# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from dataset.render import Render
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.structures import Pointclouds
from PIL import Image

from pytorch3d.loss import point_mesh_face_distance, chamfer_distance


class Evaluator:
    def __init__(self, device):

        self.render = Render(size=512, device=device)
        self.device = device

    def set_pointcloud(self, result_dict):
        self.pred = result_dict['pcd'].points_padded()
        self.gt = result_dict['verts_gt'].unsqueeze(0)

        self.tgt_mesh = self.render.VF2Mesh(result_dict['verts_gt'], result_dict['faces_gt'])

    def calculate_chamfer(self):

        chamfer_dist = chamfer_distance(self.pred, self.gt)[0] * 100.0

        src_points = Pointclouds(self.pred)
        p2s_dist = point_mesh_face_distance(self.tgt_mesh, src_points) * 100.0

        return chamfer_dist, p2s_dist

class Evaluator_mesh:
    def __init__(self, device):

        self.render = Render(size=512, device=device)
        self.device = device

    def set_mesh(self, verts_gt, faces_gt, verts_pr, faces_pr):

        self.src_mesh = self.render.VF2Mesh(verts_pr, faces_pr)
        self.tgt_mesh = self.render.VF2Mesh(verts_gt, faces_gt)

    def calculate_normal_consist(self, normal_path=None):
        self.render.meshes = self.src_mesh
        src_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3], bg='black')
        self.render.meshes = self.tgt_mesh
        tgt_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3], bg='black')

        src_normal_arr = (
            make_grid(torch.cat(src_normal_imgs, dim=0), nrow=4) + 1.0
        ) * 0.5    # [0,1]
        tgt_normal_arr = (
            make_grid(torch.cat(tgt_normal_imgs, dim=0), nrow=4) + 1.0
        ) * 0.5    # [0,1]

        error = (((src_normal_arr - tgt_normal_arr)**2).sum(dim=0).mean()) * 4.0

        if normal_path is not None:
            normal_img = Image.fromarray(
                (
                    torch.cat([src_normal_arr, tgt_normal_arr],
                            dim=1).permute(1, 2, 0).detach().cpu().numpy() * 255.0
                ).astype(np.uint8)
            )
            normal_img.save(normal_path)

        return error

    def calculate_loss(self, normal_path=None):
        self.render.meshes = self.src_mesh
        src_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3], bg='black')
        self.render.meshes = self.tgt_mesh
        tgt_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3], bg='black')

        src_normal_arr = torch.cat(src_normal_imgs, dim=0)
        tgt_normal_arr = torch.cat(tgt_normal_imgs, dim=0)
        mask = torch.logical_or((src_normal_arr.sum(1) != -3), (tgt_normal_arr.sum(1) != -3))
        diff_norm = F.cosine_similarity(src_normal_arr, tgt_normal_arr, dim=1)
        nc_dist = (diff_norm[mask] > 0.0).sum() / mask.sum()

        p2s_dist = (
            point_mesh_face_distance(self.src_mesh, Pointclouds(self.tgt_mesh.verts_packed().unsqueeze(0))) +
            point_mesh_face_distance(self.tgt_mesh, Pointclouds(self.src_mesh.verts_packed().unsqueeze(0)))
        ) * 100.0 * 0.5

        chamfer_dist = chamfer_distance(
            self.src_mesh.verts_packed().unsqueeze(0),
            self.tgt_mesh.verts_packed().unsqueeze(0),
        )[0] * 100.0

        if normal_path is not None:
            normal_img = Image.fromarray(
                (
                    (torch.cat([src_normal_arr, tgt_normal_arr],
                                dim=1).permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 * 255.0
                ).astype(np.uint8)
            )
            normal_img.save(normal_path)

        return nc_dist, chamfer_dist, p2s_dist

    def calc_acc(self, output, target, thres=0.5):

        with torch.no_grad():
            output = output.masked_fill(output < thres, 0.0)
            output = output.masked_fill(output > thres, 1.0)

            acc = output.eq(target).float().mean()

            # iou, precison, recall
            output = output > thres
            target = target > thres

            union = output | target
            inter = output & target

            _max = torch.tensor(1.0).to(output.device)

            union = max(union.sum().float(), _max)
            true_pos = max(inter.sum().float(), _max)
            vol_pred = max(output.sum().float(), _max)
            vol_gt = max(target.sum().float(), _max)

            return acc, true_pos / union, true_pos / vol_pred, true_pos / vol_gt

class EvaluatorMulti:
    def __init__(self, device):

        self.render = Render(size=512, device=device)
        self.device = device

    def set_mesh(self, points_gt, verts_pr, faces_pr):

        src_mesh = self.render.VF2Mesh(verts_pr, faces_pr)
        self.pred = sample_points_from_meshes(src_mesh, 16_384)
        self.gt = points_gt.unsqueeze(0)
    
    def set_pointcloud(self, points_gt, points_pred):
        self.pred = points_pred.points_padded()
        self.gt = points_gt.unsqueeze(0)

    def calculate_chamfer(self):

        chamfer_dist = chamfer_distance(self.pred, self.gt)[0] * 100.0

        return chamfer_dist