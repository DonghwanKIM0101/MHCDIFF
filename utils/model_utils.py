import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform


def default(x, d):
    return d if x is None else x


def get_num_points(x: Pointclouds, /):
    return x.points_padded().shape[1]


def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


class Embedder:
    def __init__(self, num_freqs):
        self.num_freqs = num_freqs

        embed_fns = []
        d = 1 # input_dims
        out_dim = 0

        # include input
        embed_fns.append(lambda x: x)
        out_dim += d

        freq_bands = 2. ** torch.linspace(0., num_freqs-1, num_freqs)

        for freq in freq_bands:
            embed_fns.append(lambda x: torch.sin(x * freq))
            out_dim += d
            embed_fns.append(lambda x: torch.cos(x * freq))
            out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)