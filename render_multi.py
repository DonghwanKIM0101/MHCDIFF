import argparse
import os
import trimesh
import numpy as np
import math
import pickle
import cv2
import torch
from tqdm import tqdm

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.io import IO


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="MultiHuman", help='dataset name')
    parser.add_argument('-type', '--type', type=str, default="single", help='type name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./data", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=3, help='number of views')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')
    args = parser.parse_args()

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views/{args.type}"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    subjects = np.loadtxt(f"./data/{args.dataset}/{args.type}.txt", dtype=str)


    # renderer = Renderer(512, device)
    device = 'cuda:0'
    scale = 100.0
    up_axis = 1
    num_views = args.num_views
    img_size = args.size


    with open('./data/smpl_related/models/smplx/SMPLX_NEUTRAL.pkl', 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
        J_regressor = smpl_data['J_regressor']

    for subject in tqdm(subjects):
        mesh_file_list = []
        smplx_file_list = []
        seg_file_list = []
        if args.type == 'single':
            mesh_file = os.path.join('data', args.dataset, args.type, 'obj', f'{subject}', f'{subject}.obj')
            smplx_file = os.path.join('data', args.dataset, args.type, 'smplx', f'{subject}', 'smplx.obj')
            num_subjects = 1
            mesh_file_list.append(mesh_file)
            smplx_file_list.append(smplx_file)
            seg_file_list.append(None)
        else:
            if 'two' in args.type:
                num_subjects = 2
            elif 'three' in args.type:
                num_subjects = 3
            else:
                num_subjects = 1
            
            for idx in range(num_subjects):

                mesh_file = os.path.join('data', args.dataset, args.type, 'obj_all', f'{subject}', f'{subject}.obj')
                smplx_file = os.path.join('data', args.dataset, args.type, 'smplx', f'{subject}_{idx}', 'smplx.obj')
                seg_file = os.path.join('data', args.dataset, args.type, 'obj', f'{subject}_{idx}', f'{subject}_{idx}.obj')

                mesh_file_list.append(mesh_file)
                smplx_file_list.append(smplx_file)
                seg_file_list.append(seg_file)


        for idx in range(num_subjects):
            mesh_file = mesh_file_list[idx]
            smplx_file = smplx_file_list[idx]
            seg_file = seg_file_list[idx]

            smplx = trimesh.load(smplx_file)
            smpl_verts = np.asarray(smplx.vertices, dtype=np.float32)
            smpl_joints = np.matmul(J_regressor, smpl_verts)

            scan = trimesh.load(mesh_file)
            verts = torch.tensor(scan.vertices).float().to(device)
            faces = torch.tensor(scan.faces).float().to(device)
            texture = torch.tensor(scan.visual.vertex_colors).float().to(device)[...,:3]
            mesh = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
            mesh.textures = TexturesVertex(verts_features=texture.unsqueeze(0))

            if seg_file is not None:
                seg_scan = trimesh.load(seg_file)
                seg_verts = torch.tensor(seg_scan.vertices).float().to(device)
                seg_faces = torch.tensor(seg_scan.faces).float().to(device)
                seg_texture = torch.tensor(seg_scan.visual.vertex_colors).float().to(device)[...,:3]
                seg_mesh = Meshes(seg_verts.unsqueeze(0), seg_faces.unsqueeze(0))
                seg_mesh.textures = TexturesVertex(verts_features=seg_texture.unsqueeze(0))
            else:
                seg_mesh = None

            center = torch.from_numpy(smpl_joints[0]).to(device)
            if seg_mesh is not None:
                center[up_axis] = 0.5 * (seg_verts[:,up_axis].max() + seg_verts[:,up_axis].min())
            else:
                center[up_axis] = 0.5 * (verts[:,up_axis].max() + verts[:,up_axis].min())
            mesh.offset_verts_(-center)
            if seg_mesh is not None:
                seg_mesh.offset_verts_(-center)
            smpl_verts = smpl_verts - center.cpu().numpy()

            elev = torch.linspace(0, 0, num_views)
            azim = torch.linspace(0, 240, num_views)
            azim += 180

            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

            R, T = look_at_view_transform(dist=100.0, elev=elev, azim=azim)
            cameras = FoVOrthographicCameras(
                R=R,
                T=T,
                znear=100.0,
                zfar=-100.0,
                max_y=100.0,
                min_y=-100.0,
                max_x=100.0,
                min_x=-100.0,
                scale_xyz=(scale * np.ones(3), ),
                device=device
            )

            raster_settings = RasterizationSettings(
                image_size=img_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                max_faces_per_bin=100000,
            )


            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(
                    device=device
                )
            )

            meshes = mesh.extend(num_views)
            target_images = renderer(meshes, cameras=cameras, lights=lights)
            target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
            target_cameras = [FoVOrthographicCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]
            
            if seg_mesh is not None:
                seg_meshes = seg_mesh.extend(num_views)
                target_images_seg = renderer(seg_meshes, cameras=cameras, lights=lights)
                target_rgb_seg = [target_images_seg[i, ..., :3] for i in range(num_views)]

            if 'single' not in args.type:
                save_folder = os.path.join(current_out_dir, f'{subject}_{idx}')
            else:
                save_folder = os.path.join(current_out_dir, subject)
            os.makedirs(os.path.join(save_folder, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'scan'), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'smpl'), exist_ok=True)

            if seg_mesh is None:
                verts = mesh.verts_packed()
                faces = scan.faces
            else:
                verts = seg_mesh.verts_packed()
                faces = seg_scan.faces

            for i, rot in enumerate(azim):
                rot = int(rot)
                R = make_rotate(0, math.radians(-rot), 0)
                rot = rot - 180
                R_tensor = torch.from_numpy(R).float().to(device)
                
                all_rgb = target_rgb[i].cpu().numpy().astype(np.uint8)[...,::-1]
                cv2.imwrite(os.path.join(save_folder, 'rgb', f'{rot:03d}_all.png'), all_rgb)

                if seg_mesh is None:
                    cv2.imwrite(os.path.join(save_folder, 'rgb', f'{rot:03d}.png'), all_rgb)
                else:
                    seg_rgb = target_rgb_seg[i].cpu().numpy().astype(np.uint8)[...,::-1]
                    # mask = ((all_rgb == seg_rgb).sum(-1) != 3)
                    mask = ((all_rgb - seg_rgb).sum(-1) > 15)
                    all_rgb[mask] = 1
                    cv2.imwrite(os.path.join(save_folder, 'rgb', f'{rot:03d}.png'), all_rgb)

                verts_rot = torch.mm(R_tensor, verts.T).T
                trimesh.Trimesh(verts_rot.cpu().numpy(), faces).export(
                    os.path.join(save_folder, 'scan', f'{rot:03d}.obj')
                )

                smpl_verts_rot = np.matmul(R, smpl_verts.T).T
                trimesh.Trimesh(smpl_verts_rot, smplx.faces).export(
                    os.path.join(save_folder, 'smpl', f'{rot:03d}.obj')
                )
            