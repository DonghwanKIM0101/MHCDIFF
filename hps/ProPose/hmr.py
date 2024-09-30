import torch
from einops import rearrange

from propose.models import builder
from propose.utils.config import update_config

class HMR():
    def __init__(
            self,
            device,
            cfg_file,
            pretrained_checkpoint,
        ):

        cfg = update_config(cfg_file)

        self.hmr_model = builder.build_model(cfg.MODEL)
        model_state = self.hmr_model.state_dict()
        pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        matched_pretrained_state = {k: v for k, v in pretrained_state.items()
                                    if not k.startswith('smpl.')}
        
        model_state.update(matched_pretrained_state)
        self.hmr_model.load_state_dict(model_state)

        self.hmr_model.to(device)
        self.hmr_model.eval()

    def forward(self, x):

        bSize = x.shape[0]
        pose_output = self.hmr_model(x, flip_test=False, use_sample=False)

        smpl_vertices = pose_output['pred_vertices'].unsqueeze(1)
        scale = pose_output['pred_cam'][:, 0]
        trans = torch.cat([pose_output['pred_cam'][:, 1:], torch.zeros_like(pose_output['pred_cam'])[:,[-1]]], dim=-1)

        pose_output = self.hmr_model(x, flip_test=False, use_sample=True)
        sample_vertices = pose_output['pred_vertices']
        sample_vertices = rearrange(sample_vertices, '(b s) n d -> b s n d', b=bSize)

        smpl_vertices = torch.cat([smpl_vertices, sample_vertices], dim=1)
        smpl_vertices = (smpl_vertices + trans[:, None, None, :]) * scale[..., None, None, None]
        smpl_vertices[..., 1:] *= -1.

        return smpl_vertices