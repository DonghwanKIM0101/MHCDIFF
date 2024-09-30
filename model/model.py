import inspect
from typing import Optional

import os
import sys
import random
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from tqdm import tqdm

from utils.model_utils import get_custom_betas
from .point_cloud_model import PointCloudModel
from .projection_model import PointCloudProjectionModel
from .camera import get_camera

from utils.mesh_util import SMPLX
from hps.pixielib.utils.config import cfg as pixie_cfg
from hps.pixielib.pixie import PIXIE

sys.path.append('./hps/ProPose')
from hps.ProPose.hmr import HMR


class ConditionalPointCloudDiffusionModel(PointCloudProjectionModel):
    
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)
        
        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels
        )

    def prepare_smpl(self):
        if self.use_smpl_features:

            if self.predict_smpl == 'pixie':
                self.hps = PIXIE(config=pixie_cfg, device=self.device)
                self.smpl_model = self.hps.smplx
                self.smpl_faces = self.smpl_model.faces_tensor
                smpl_data = SMPLX()
                self.smpl_faces = self.smpl_faces[~smpl_data.smplx_eyeball_fid]
                mouth_face = torch.as_tensor(smpl_data.smplx_mouth_fid).to(self.device)
                self.smpl_faces = torch.cat([self.smpl_faces, mouth_face], dim=0)

            # elif self.predict_smpl == 'propose':
            elif self.predict_smpl in ['propose', 'pymaf']:
                self.hps = HMR(
                    device=self.device,
                    cfg_file=os.path.join(os.path.dirname(__file__), '../hps/ProPose/configs/smpl_hm_xyz.yaml'),
                    pretrained_checkpoint=os.path.join(os.path.dirname(__file__), '../hps/ProPose/model_files/propose_hr48_xyz.pth')
                )
                self.smpl_faces = self.hps.hmr_model.smpl.faces_tensor

    @torch.no_grad()
    def smpl_estimate(
        self,
        image: Tensor,
        num_samples: int = 1,
    ):
        if self.predict_smpl == 'pixie':
            preds_dict = self.hps.forward(image)

            smpl_verts, _, _ = self.smpl_model(
                shape_params=preds_dict['shape'],
                expression_params=preds_dict['exp'],
                body_pose=preds_dict['body_pose'],
                global_pose=preds_dict['global_pose'],
                jaw_pose=preds_dict['jaw_pose'],
                left_hand_pose=preds_dict['left_hand_pose'],
                right_hand_pose=preds_dict['right_hand_pose'],
            )

            scale = preds_dict['cam'][:, 0]
            trans = torch.cat([preds_dict['cam'][:, 1:], torch.zeros_like(preds_dict['cam'])[:,[-1]]], dim=-1)

            smpl_verts = (smpl_verts + trans.unsqueeze(1)) * scale[...,None,None]
            smpl_verts[..., 1:] *= -1.

            smpl_verts = smpl_verts.unsqueeze(1)
            return smpl_verts

        elif self.predict_smpl == 'propose':
            smpl_verts = self.hps.forward(image)
            return smpl_verts[:, :num_samples]

    def forward_train(
        self, 
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        smpl_feats: Optional[dict],
        return_intermediate_steps: bool = False 
    ):

        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc)
        B, N, D = x_0.shape

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), 
            device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Conditioning
        x_t_input = self.get_input_with_conditioning(x_t, camera=camera, 
            image_rgb=image_rgb, mask=mask, smpl_feats=smpl_feats, t=timestep)

        # Forward
        noise_pred = self.point_cloud_model(x_t_input, timestep)
        
        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss

    @torch.no_grad()
    def forward_sample(
        self,
        num_points: int,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        smpl_feats: Optional[dict],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Return SMPL
        return_smpl: bool = True,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = 1 if image_rgb is None else image_rgb.shape[0]
        D = 3 + (self.color_channels if self.predict_color else 0)
        device = self.device if image_rgb is None else image_rgb.device
        
        # Sample noise
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)
        for i, t in enumerate(progress_bar):
            
            # Conditioning
            x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
                image_rgb=image_rgb, mask=mask, smpl_feats=smpl_feats, t=t)
            
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(x_t)
        result_dict = {
            'output': output
        }
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o) for o in all_outputs]
            result_dict['all_outputs'] = all_outputs
        
        if return_smpl:
            smpl = self.tensor_to_point_cloud(smpl_feats['smpl_verts'][:, 0])
            result_dict['smpl'] = smpl
            result_dict['smpl_faces'] = smpl_feats['smpl_faces'][:, 0]

            result_dict['sample_smpl'] = smpl_feats['smpl_verts']
        
        return result_dict

    def forward(self, batch: dict, mode: str = 'train', **kwargs):

        camera = get_camera(batch['image'].device)

        if self.predict_smpl == 'gt':
            smpl_verts = batch['smplx_verts'].unsqueeze(1)
            smpl_faces = batch['smplx_faces'].unsqueeze(1)
            smpl_norm = batch['smplx_norms'].unsqueeze(1)
        else:
            if self.predict_smpl == 'propose':
                num_samples = 10
            else:
                num_samples = 1
            smpl_verts = self.smpl_estimate(batch['img_hps'], num_samples)
            bSize = smpl_verts.shape[0]
            smpl_faces = self.smpl_faces[None, None, ...].repeat(bSize, num_samples, 1, 1)
            smpl_norm = None

        smpl_feats = {
            'smpl_verts': smpl_verts,
            'smpl_faces': smpl_faces,
            'smpl_norms': smpl_norm
        }

        mask = (torch.mean(batch['image'], dim=1, keepdim=True) != 0.0).float()

        """A wrapper around the forward method for training and inference"""
        if mode == 'train':
            pc = self.tensor_to_point_cloud(batch['points'])
            return self.forward_train(
                pc=pc, 
                camera=camera, 
                image_rgb=batch['image'],
                mask=mask,
                smpl_feats=smpl_feats,
                **kwargs) 
        elif mode == 'sample':
            num_points = batch['num_points'][0]
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=batch['image'],
                mask=mask,
                smpl_feats=smpl_feats,
                **kwargs) 
        else:
            raise NotImplementedError()