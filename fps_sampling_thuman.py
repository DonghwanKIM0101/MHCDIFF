import argparse
import os
import numpy as np
import trimesh
import torch
from tqdm import tqdm


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FPS THuman raw scans')
    parser.add_argument(
        '--src_dataset_path',
        type=str,
        default='data/thuman2',
        help='Path to THuman dataset.'
    )
    parser.add_argument(
        '--dst_dataset_path',
        type=str,
        default='data/thuman2_sampling',
        help='Directory path to store FPS data.',
    )
    args = parser.parse_args()
    # num_samples = [8_192, 16_384, 32_768, 65_536, 131_072]
    num_samples = [16_384]
    dst_dataset_path_list = []
    for num_sample in num_samples:
        dst_dataset_path = args.dst_dataset_path + f'_{num_sample}'
        os.makedirs(dst_dataset_path, exist_ok=True)
        dst_dataset_path_list.append(dst_dataset_path)

    subject_list = np.loadtxt(os.path.join(args.src_dataset_path, 'all.txt'), dtype=str).tolist()

    for subject in tqdm(subject_list):
        scan_path = os.path.join(args.src_dataset_path, f'scans/{subject}/{subject}.obj')
        scan = trimesh.load(scan_path)
        scan_vertices = np.asarray(scan.vertices)
        scan_vertices = torch.Tensor(scan_vertices).cuda().unsqueeze(0)

        for i, num_sample in enumerate(num_samples):
            sampled_points = index_points(
                scan_vertices,
                farthest_point_sample(scan_vertices, num_sample)
            )

            sampled_points = sampled_points[0].cpu().numpy()

            np.save(os.path.join(dst_dataset_path_list[i], f'{subject}.npy'), sampled_points)
