import datetime
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF

from utils.mesh_util import evaluate_registration
import utils.training_utils as training_utils
import utils.diffusion_utils as diffusion_utils
from dataset import get_dataset
from dataset.Evaluator import Evaluator, Evaluator_mesh, EvaluatorMulti
from model import get_model, ConditionalPointCloudDiffusionModel
from config.structured import ProjectConfig

try:
    import lovely_tensors
    lovely_tensors.monkey_patch()
except ImportError:
    pass  # lovely tensors is not necessary but it really is lovely, I do recommend it

import warnings
warnings.filterwarnings("ignore") # torch.meshgrid

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):

    # Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, 
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs])

    # Logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.job, 
                   config=OmegaConf.to_container(cfg))
        wandb.run.log_code(root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
            exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
        cfg: ProjectConfig = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    # Model
    model = get_model(cfg)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # Datasets
    dataloader_train, dataloader_val, dataloader_test = get_dataset(cfg)
    
    # Compute total training batch size
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

    # Setup. Note that this does not currently work with CO3D.
    model, optimizer, scheduler, dataloader_train, dataloader_test, dataloader_val = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_test, dataloader_val)
    
    model.prepare_smpl()

    # Type hints
    model: ConditionalPointCloudDiffusionModel
    optimizer: torch.optim.Optimizer

    # Visualize before training
    if cfg.run.job == 'vis':
        visualize(
            cfg=cfg,
            model=model,
            dataloader_val=None,
            dataloader_vis=dataloader_test,
            accelerator=accelerator,
            identifier='init',
        )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
            time.sleep(5)
        return

    if cfg.run.vis_before_training:
        visualize(
            cfg=cfg,
            model=model,
            dataloader_val=None,
            dataloader_vis=dataloader_test,
            accelerator=accelerator,
            identifier='init',
            num_batches=1, 
        )

    # Sample from the model
    if cfg.run.job == 'sample':
        # Whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert model_ema is not None
            model_ema.to(accelerator.device)
            sample_context = model_ema.average_parameters
        else:
            sample_context = nullcontext
        # Sample
        with sample_context():
            sample(
                cfg=cfg,
                model=model,
                dataloader=dataloader_test,
                accelerator=accelerator,
            )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataloader_train.dataset):_}')
    print(f'    Dataset val size: {len(dataloader_val.dataset):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader test size: {len(dataloader_test):_}')
    print(f'    Batch size per device = {cfg.dataloader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
    print(f'    Max training steps = {cfg.run.max_steps}')
    print(f'    Training state = {train_state}')

    # Infinitely loop training
    while True:
    
        # Train progress bar
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, 
            header=log_header)

        # Train
        for i, batch in enumerate(progress_bar):
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            model.train()

            # Gradient accumulation
            with accelerator.accumulate(model):

                # Forward
                loss = model(batch, mode='train')

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # grad_norm_unclipped = training_utils.compute_grad_norm(model.parameters())  # useless w/ mixed prec
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1

                # Exit if loss was NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            # Gradient accumulation
            if accelerator.sync_gradients:

                # Logging
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    # 'grad_norm_unclipped': grad_norm_unclipped,  # useless w/ mixed prec
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if (cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                    wandb.log(log_dict, step=train_state.step)
            
                # Update EMA
                if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                    model_ema.update(model.parameters())

                # Save a checkpoint
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'model_ema': model_ema.state_dict() if model_ema else {},
                        'cfg': cfg
                    }
                    checkpoint_path = 'checkpoint-latest.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

                # Visualize
                if (cfg.run.vis_freq > 0) and (train_state.step % cfg.run.vis_freq) == 0:
                    visualize(
                        cfg=cfg,
                        model=model,
                        dataloader_val=dataloader_val,
                        dataloader_vis=dataloader_test,
                        accelerator=accelerator,
                        identifier=f'{train_state.step}', 
                        num_batches=1,
                        train_step=train_state.step,
                    )

                # End training after the desired number of steps/epochs
                if train_state.step >= cfg.run.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    
                    wandb.finish()
                    time.sleep(5)
                    return

        # Epoch complete, log it and continue training
        train_state.epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)


@torch.no_grad()
def visualize(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader_val: Optional[Iterable],
    dataloader_vis: Iterable,
    accelerator: Accelerator,
    identifier: str = '',
    num_batches: Optional[int] = None,
    output_dir: str = 'vis',
    train_step: Optional[int] = None,
):
    from model.camera import get_camera
    from pytorch3d.vis.plotly_vis import plot_scene
    from pytorch3d.structures import Pointclouds

    # Eval mode
    model.eval()
    metric_logger = training_utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_vis, cfg.run.print_step_freq, "Vis")

    # Validation
    if dataloader_val is not None:
        progress_bar_val = metric_logger.log_every(dataloader_val, cfg.run.print_step_freq, "Val")

        total_loss = 0.
        num_loss = 0
        for batch_idx, batch in enumerate(progress_bar_val):
            if train_step is not None:
                loss = model(batch, mode='train')
                b_size = len(batch['name'])
                total_loss += b_size * loss.item()
                num_loss += b_size

    # Output dir
    output_dir: Path = Path(output_dir)
    (output_dir / 'raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'pointclouds').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images').mkdir(exist_ok=True, parents=True)
    (output_dir / 'rotate').mkdir(exist_ok=True, parents=True)
    (output_dir / 'videos').mkdir(exist_ok=True, parents=True)
    (output_dir / 'evolutions').mkdir(exist_ok=True, parents=True)

    # Visualize
    wandb_log_dict = {}
    for batch_idx, batch in enumerate(progress_bar):

        if num_batches is not None and batch_idx >= num_batches:
            continue

        if cfg.dataset.type in ['hi4d', 'multihuman']:
            batch['image'] = torch.cat([img.unsqueeze(0) for img in batch['image']], dim=0)
            batch['img_hps'] = torch.cat([img.unsqueeze(0) for img in batch['img_hps']], dim=0)

        # Sample
        result_dict = model(batch, mode='sample', return_sample_every_n_steps=100, 
            num_inference_steps=cfg.run.num_inference_steps, disable_tqdm=(not accelerator.is_main_process))
        output: Pointclouds = result_dict['output']
        all_outputs: List[Pointclouds] = result_dict['all_outputs']
        # list of B Pointclouds, each with a batch size of return_sample_every_n_steps
        smpl: Pointclouds = result_dict['smpl']

        sample_smpl = result_dict['sample_smpl']
        
        # Filenames
        filestr = str(output_dir / '{dir}' / f'p-{accelerator.process_index}-b-{batch_idx}-s-{{i:02d}}-{{name}}-{identifier}.{{ext}}')
        filestr_wandb = f'{{dir}}/b-{batch_idx}-{{name}}-s-{{i:02d}}-{{name}}'

        # Save raw samples
        filename = filestr.format(dir='raw', name='raw', i=0, ext='pth')
        torch.save({'output': output, 'all_outputs': all_outputs, 'batch': batch}, filename)

        # Save individual samples
        for i in range(len(output)):
            camera = get_camera(accelerator.device)
            
            if 'verts' in batch.keys():
                gt_pointcloud = Pointclouds(batch['verts'][i].unsqueeze(0))[0]
            else:
                gt_pointcloud = Pointclouds(batch['points'][[i]])[0]
            smpl_pointcloud = smpl[i]
            pred_pointcloud = output[i]
            pred_all_pointclouds = all_outputs[i]

            sample_pointcloud = Pointclouds(points=sample_smpl[i])
            num_samples = sample_smpl[i].shape[0]

            # Plot using plotly and pytorch3d
            fig = plot_scene({ 
                # 'SMPL': {'pointcloud': smpl_pointcloud},
                'SMPL': {
                    f'{idx}': sample_pointcloud[idx] for idx in range(num_samples)
                },
                'Pred': {'pointcloud': pred_pointcloud},
                'GT': {'pointcloud': gt_pointcloud},
            }, ncols=2, viewpoint_cameras=camera, pointcloud_max_points=16_384)
            
            # Save plot
            filename = filestr.format(dir='pointclouds', name='pointclouds', i=i, ext='html')
            fig.write_html(filename)

            # Add to W&B
            filename_wandb = filestr_wandb.format(dir='pointclouds', name='pointclouds', i=i)
            wandb_log_dict[filename_wandb] = wandb.Html(open(filename), inject=False)

            # Save input images
            filename = filestr.format(dir='images', name='image_rgb', i=i, ext='png')
            TVF.to_pil_image(batch['image'][i]).save(filename)

            # Add to W&B
            filename_wandb = filestr_wandb.format(dir='images', name='image_rgb', i=i)
            wandb_log_dict[filename_wandb] = wandb.Image(filename)

            # Loop
            for name, pointcloud in (('gt', gt_pointcloud), ('pred', pred_pointcloud), ('smpl', smpl_pointcloud)):
            
                # Render gt/pred point cloud from given view
                filename_image = filestr.format(dir='images', name=name, i=i, ext='png')
                filename_image_wandb = filestr_wandb.format(dir='images', name=name, i=i)
                diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud, 
                    output_file_image=filename_image, cameras=camera, scale_factor=cfg.model.scale_factor)
                wandb_log_dict[filename_image_wandb] = wandb.Image(filename_image)

                # Render gt/pred point cloud from rotating view
                filename_video = filestr.format(dir='videos', name=name, i=i, ext='gif')
                filename_video_wandb = filestr_wandb.format(dir='videos', name=name, i=i)
                diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud, 
                    output_file_video=filename_video, num_frames=30, scale_factor=cfg.model.scale_factor)
                wandb_log_dict[filename_video_wandb] = wandb.Video(filename_video)

                # Render gt/pred point cloud from rotating view
                filename_rotate = filestr.format(dir='rotate', name=name, i=i, ext='png')
                filename_rotate_wandb = filestr_wandb.format(dir='rotate', name=name, i=i)
                diffusion_utils.visualize_pointcloud_batch_pytorch3d_rotate(pointclouds=pointcloud, 
                    output_file_image=filename_rotate, scale_factor=cfg.model.scale_factor)
                wandb_log_dict[filename_rotate_wandb] = wandb.Image(filename_rotate)

            # Render point cloud diffusion evolution
            filename_evo = filestr.format(dir='evolutions', name='evolutions', i=i, ext='gif')
            filename_evo_wandb = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
            diffusion_utils.visualize_pointcloud_evolution_pytorch3d(
                pointclouds=pred_all_pointclouds, output_file_video=filename_evo, camera=camera)
            wandb_log_dict[filename_evo_wandb] = wandb.Video(filename_evo)

    # Save to W&B
    if cfg.logging.wandb and accelerator.is_local_main_process:
        wandb.log(wandb_log_dict, commit=False)
        if train_step is not None:
            wandb.log({'validation_loss': total_loss / num_loss}, step=train_step)

    print('Saved visualizations to: ')
    print(output_dir.absolute())


@torch.no_grad()
def sample(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = 'sample',
):
    from pytorch3d.io import IO
    from pytorch3d.structures import Pointclouds
    from tqdm import tqdm

    # Eval mode
    model.eval()
    progress_bar = tqdm(dataloader, disable=(not accelerator.is_main_process))

    # Output dir
    output_dir: Path = Path(output_dir)

    # PyTorch3D IO
    io = IO()

    # Evaluator
    if cfg.dataset.type == 'multihuman':
        evaluator = EvaluatorMulti(device=accelerator.device)
    else:
        evaluator = Evaluator(device=accelerator.device)
        evaluator_mesh = Evaluator_mesh(device=accelerator.device)
    loss_dict = {
        'chamfer': [],
        'p2s': [],
        'nc': [],
        'chamfer_smpl': [],
        'p2s_smpl': [],
        'nc_smpl': [],
    }

    # Visualize
    for batch_idx, batch in enumerate(progress_bar):
        if cfg.dataset.type != 'demo':
            gt_verts = [verts.unsqueeze(0) for verts in batch['verts']]
            gt_faces = [faces.unsqueeze(0) for faces in batch['faces']]
        if cfg.dataset.type in ['hi4d', 'multihuman']:
            batch['image'] = torch.cat([img.unsqueeze(0) for img in batch['image']], dim=0)
            batch['img_hps'] = torch.cat([img.unsqueeze(0) for img in batch['img_hps']], dim=0)

        # progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')
        if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
            break

        # Optionally produce multiple samples for each point cloud
        for sample_idx in range(cfg.run.num_samples):

            # Filestring
            filename = f'{{name}}-{sample_idx}.{{ext}}' if cfg.run.num_samples > 1 else '{name}.{ext}'
            filestr = str(output_dir / '{dir}' / filename)

            # Sample
            result_dict = model(batch, mode='sample', return_sample_every_n_steps=10, scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps, disable_tqdm=(not accelerator.is_main_process))
            output: Pointclouds = result_dict['output']
            all_outputs: List[Pointclouds] = result_dict['all_outputs']
            # list of B Pointclouds, each with a batch size of return_sample_every_n_steps
            smpl: Pointclouds = result_dict['smpl']
            smpl_verts = smpl.points_padded()
            smpl_faces = result_dict['smpl_faces']

            # Save individual samples
            for i in range(len(output)):
                sequence_name = batch['name'][i]
                (output_dir / 'gt').mkdir(exist_ok=True, parents=True)
                (output_dir / 'pred').mkdir(exist_ok=True, parents=True)
                (output_dir / 'smpl').mkdir(exist_ok=True, parents=True)
                (output_dir / 'images').mkdir(exist_ok=True, parents=True)
                (output_dir / 'evolutions').mkdir(exist_ok=True, parents=True)
                
                if cfg.dataset.type != 'demo':
                    # Save ground truth
                    pc = Pointclouds(gt_verts[i])[0]
                    io.save_pointcloud(data=pc, path=filestr.format(dir='gt', 
                        name=sequence_name, ext='ply'))
                
                # Save generation
                io.save_pointcloud(data=output[i], path=filestr.format(dir='pred', 
                    name=sequence_name, ext='ply'))
                
                # Save smpl
                io.save_pointcloud(data=smpl[i], path=filestr.format(dir='smpl', 
                    name=sequence_name, ext='ply'))

                # Save input images
                filename = filestr.format(dir='images', name=sequence_name, ext='png')
                TVF.to_pil_image(batch['image'][i]).save(filename)

                # Save evolutions
                if cfg.run.sample_save_evolutions:
                    torch.save(all_outputs[i], filestr.format(dir='evolutions', 
                        name=sequence_name, ext='pth'))
                
                if cfg.dataset.type != 'demo':

                    if cfg.dataset.type == 'multihuman':
                        pred_pcd = output[i]
                        pred_smpl = smpl_verts[i]

                        evaluator.set_pointcloud(
                            points_gt=gt_verts[i][0],
                            points_pred=pred_pcd,
                        )
                        chamfer = evaluator.calculate_chamfer()
                        p2s = 0.0
                        nc = 0.0

                        evaluator.set_mesh(
                            points_gt=gt_verts[i][0],
                            verts_pr=pred_smpl,
                            faces_pr=smpl_faces[i],
                        )
                        chamfer_smpl = evaluator.calculate_chamfer()
                        p2s_smpl = 0.0
                        nc_smpl = 0.0
                    
                    else:
                        if cfg.dataset.type == 'hi4d':
                            verts_gt = gt_verts[i][0].cpu().numpy()
                            verts_pred = output[i].points_padded()[0].cpu().numpy()
                            verts_smpl = smpl_verts[i].cpu().numpy()
                            
                            verts_pred, verts_smpl = evaluate_registration(verts_gt, verts_pred, verts_smpl)

                            verts_pred = torch.tensor(verts_pred).float().to(accelerator.device)
                            pred_pcd = Pointclouds(points=verts_pred.unsqueeze(0))
                            pred_smpl = torch.tensor(verts_smpl).float().to(accelerator.device)
                        
                        else:
                            pred_pcd = output[i]
                            pred_smpl = smpl_verts[i]

                        # Evaluate
                        result_eval = {
                            "verts_gt": gt_verts[i][0],
                            "faces_gt": gt_faces[i][0],
                            "pcd": pred_pcd,
                        }
                        evaluator.set_pointcloud(result_eval)
                        chamfer, p2s = evaluator.calculate_chamfer()
                        nc = 0.0
                        
                        # Evaluate SMPL
                        evaluator_mesh.set_mesh(
                            verts_gt=gt_verts[i][0],
                            faces_gt=gt_faces[i][0],
                            verts_pr=pred_smpl,
                            faces_pr=smpl_faces[i],
                        )
                        nc_smpl, chamfer_smpl, p2s_smpl = evaluator_mesh.calculate_loss()
                
                    loss_dict['chamfer'].append(chamfer)
                    loss_dict['p2s'].append(p2s)
                    loss_dict['nc'].append(0.0)

                    loss_dict['chamfer_smpl'].append(chamfer_smpl)
                    loss_dict['p2s_smpl'].append(p2s_smpl)
                    loss_dict['nc_smpl'].append(nc_smpl)

                    # Logging
                    progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d} | chamfer: {chamfer:0.3f}, p2s: {p2s:0.3f}')

    print('Saved samples to: ')
    print(output_dir.absolute())

    if cfg.dataset.type != 'demo':
        chamfer = torch.tensor(loss_dict['chamfer'])
        p2s = torch.tensor(loss_dict['p2s'])
        nc = torch.tensor(loss_dict['nc'])

        print('evaluation: ')
        print(f'chamfer: {chamfer.mean().item():.3f}')
        print(f'p2s: {p2s.mean().item():.3f}')
        print(f'nc: {nc.mean().item():.3f}')

        chamfer_smpl = torch.tensor(loss_dict['chamfer_smpl'])
        p2s_smpl = torch.tensor(loss_dict['p2s_smpl'])
        nc_smpl = torch.tensor(loss_dict['nc_smpl'])

        print('evaluation SMPL: ')
        print(f'chamfer: {chamfer_smpl.mean().item():.3f}')
        print(f'p2s: {p2s_smpl.mean().item():.3f}')
        print(f'nc: {nc_smpl.mean().item():.3f}')

if __name__ == '__main__':
    main()