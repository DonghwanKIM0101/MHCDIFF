import os
import glob
import cv2
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.im_utils import *
from utils.mesh_util import *
from config.structured import THuman2Config

class THuman(data.Dataset):
    def __init__(self, cfg: THuman2Config, split='train', job='train'):
        
        self.split = split
        self.job = job
        self.root = cfg.root
        
        self.dataset = cfg.type
        self.input_size = cfg.image_size
        self.scale = cfg.scale_factor
        self.rotation_num = cfg.rotation_num
        self.max_points = cfg.max_points

        random_erasing = cfg.random_erasing
        self.hps = cfg.hps

        if self.split != 'train':
            self.rotations = range(0, 360, 120)
        else:
            self.rotations = range(0, 360, 360 // self.rotation_num)
        
        if self.dataset == 'multihuman':
            self.rotation_num = 3
            random_erasing = False
            self.category = cfg.category
            if 'two' in self.category:
                self.num_identities = 2
            elif 'three' in self.category:
                self.num_identities = 3
            else:
                self.num_identities = 1
        elif self.dataset == 'hi4d':
            self.num_identities = 1
            self.rotations = [0]
            random_erasing = False
        elif self.dataset == 'demo':
            self.num_identities = 1
            self.rotations = [0]
            random_erasing = False
        else:
            self.num_identities = 1
        
            if self.split == 'test' and self.job in ['sample', 'vis']:
                self.dataset = 'cape'
                self.rotation_num = 3
                random_erasing = False
        
        self.feat_keys = ["smpl_verts", "smpl_faces"]

        dataset_dir = os.path.join(self.root, 'data', self.dataset)
        self.points_dir = dataset_dir + '_sampling_16384'
        self.mesh_dir = os.path.join(dataset_dir, 'scans')
        self.smplx_dir = os.path.join(dataset_dir, 'smplx')
        self.smpl_dir = os.path.join(dataset_dir, 'smpl')

        self.subject_list = self.get_subject_list(split)

        self.smplx = SMPLX()

        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5 if random_erasing else 0., scale=(0.0,0.4)),
        ])

        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0, ), (1.0, ))
        ])

        if self.hps == 'pixie':
            self.image_to_hps_tensor = transforms.Compose(
                [
                    transforms.Resize(224)
                ]
            )
        
        elif self.hps == 'propose':
            self.image_to_hps_tensor = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]
            )

    def get_subject_list(self, split):
        if self.dataset in ['thuman2', 'cape']:
            subject_list = []
            split_txt = os.path.join(self.root, 'data', self.dataset, f'{split}.txt')
            subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        elif self.dataset == 'hi4d':
            subject_list = []
            seq_list = np.loadtxt(osp.join(self.root, 'Hi4D', 'test.txt'), dtype=str)

            for seq_path in seq_list:
                subject_list += list(
                    glob.glob(
                        osp.join(
                            self.root, 'data', seq_path.replace('Hi4D', 'Hi4D_undist'), 'data', "*.npz"
                        )
                    )
                )
            
        elif self.dataset == 'multihuman':
            subject_list = []
            split_txt = os.path.join(self.root, 'data', 'MultiHuman', f'{self.category}.txt')
            subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        elif self.dataset == 'demo':
            subject_list = list(
                glob.glob(
                    osp.join(
                        self.root, 'data', 'demo', 'images', '*.png'
                    )
                )
            )

        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations) * self.num_identities
    
    def _getitem_thuman(self, index):
        data_dict = {}

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset_folder = os.path.join(self.root, 'data', f'{self.dataset}_{self.rotation_num}views', subject)

        # setup paths
        data_dict = {
            'name': f'{self.dataset}_{subject}_{rotation:03d}',
            'subject': subject,
            'num_points': self.max_points,
            'calib_path': os.path.join(dataset_folder, 'calib', f'{rotation:03d}.txt'),
            'image_path': os.path.join(dataset_folder, 'render', f'{rotation:03d}.png'),
            'smpl_path': os.path.join(self.smpl_dir, f"{subject}.obj"),
            'vis_path': osp.join(dataset_folder, 'vis', f'{rotation:03d}.pt')
        }
        
        if self.dataset == 'thuman2':
            data_dict.update({
                'points_path': os.path.join(self.points_dir, f"{subject}.npy"),
                'mesh_path': os.path.join(self.mesh_dir, f"{subject}/{subject}.obj"),
                'smplx_path': os.path.join(self.smplx_dir, f"{subject}.obj"),
                'smpl_param': os.path.join(self.smpl_dir, f"{subject}.pkl"),
                'smplx_param': os.path.join(self.smplx_dir, f"{subject}.pkl")
            })
        
        elif self.dataset == 'cape':
            data_dict.update({
                'image_path': os.path.join(self.root, 'data', 'cape_masked', '0.2', 'images', f'cape_{subject}_{rotation:03d}.png'),
                'mesh_path': os.path.join(self.mesh_dir, f"{subject}.obj"),
                'smpl_param': os.path.join(self.smpl_dir, f"{subject}.npz")
            })
        
        # load training data
        data_dict.update(self.load_calib(data_dict['calib_path']))

        # image loader
        data_dict['image'] = self.imagepath2tensor(data_dict['image_path'])

        # image for hps
        if self.hps == 'pixie':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps
        elif self.hps == 'propose':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps
        else:
            data_dict.update(self.load_smpl(data_dict))
        
        if self.dataset == 'cape':
            data_dict.update(self.load_mesh(data_dict))
        else:
            data_dict.update(self.load_points(data_dict))

        path_keys = [key for key in data_dict.keys() if '_path' in key or '_dir' in key or '_param' in key or 'calib' in key]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    def _getitem_hi4d(self, index):
        result = []

        data_path = self.subject_list[index]
        data_dict = np.load(data_path)
        img_path = os.path.join(self.root, data_dict['image_path'].item())
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        pair_path = data_dict['pair_path'].item()
        seq_path = data_dict['seq_path'].item()
        idx = data_dict['index'].item()
        num_persons = data_dict['num_persons'].item()

        img_ori = load_img(img_path)

        gt_mask_list = []
        for instance_idx in range(num_persons):
            mask_path = data_dict[f'mask_{instance_idx}_path'].item()
            gt_mask = cv2.imread(os.path.join(self.root, mask_path), 0)
            gt_mask = cv2.resize(gt_mask, img_ori.shape[:2][::-1])
            gt_mask_list.append(gt_mask)
        
        seg_res_path = osp.join(self.root, 'data', 'Hi4D_schp', 'img_seg_mask', pair_path, seq_path, f'{idx:05}.png')
        seg_res = cv2.imread(seg_res_path)
        mask_list = [
            ((seg_res[...,1] != 128) & (seg_res[...,2] == 128)),
            ((seg_res[...,1] == 128) & (seg_res[...,2] != 128)),
            ((seg_res[...,1] == 128) & (seg_res[...,2] == 128))
        ]

        pred_mask_list = []

        for pred_idx in range(3):
            iou = 0
            seg_mask = mask_list[pred_idx]
            tgt_instance = 0

            for instance_idx in range(num_persons):
                gt_mask = gt_mask_list[instance_idx]
                cur_iou = (gt_mask & seg_mask).sum()
                if cur_iou > iou:
                    iou = cur_iou
                    pred_mask = seg_mask
                    tgt_instance = instance_idx
            
            if iou > 0:
                pred_mask_list.append((tgt_instance, pred_mask))

        calib = torch.from_numpy(data_dict['calib']).float()

        for pred_idx in range(len(pred_mask_list)):
            instance_idx, pred_mask = pred_mask_list[pred_idx]

            verts = torch.from_numpy(data_dict[f'verts_{instance_idx}']).float()
            verts = projection(verts, calib).float()
            verts = verts * torch.tensor(np.array([[1.0, -1.0, 1.0]])).float()
            faces = torch.from_numpy(data_dict[f'faces_{instance_idx}']).long()

            result_dict = {
                'name': f'{self.dataset}_{pair_path}_{seq_path}_{img_name}_{pred_idx}',
                'num_points': self.max_points,
                'verts': verts,
                'faces': faces,
            }

            img_np = process_image(
                img_ori, pred_mask, self.input_size
            )

            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray((np.mean(img_np, 2) > 0))

            img_rgb = self.image_to_tensor(img_pil)
            img_mask = torch.tensor(1.0) - (self.mask_to_tensor(mask_pil.split()[-1]) < torch.tensor(0.5)).float()
            result_dict['image'] = img_rgb * img_mask

            # image for hps
            if self.hps == 'pixie':
                image_to_hps = result_dict['image']
                image_to_hps = self.image_to_hps_tensor(image_to_hps)
                result_dict['img_hps'] = image_to_hps
            elif self.hps == 'propose':
                image_to_hps = result_dict['image']
                image_to_hps = self.image_to_hps_tensor(image_to_hps)
                result_dict['img_hps'] = image_to_hps

            result.append(result_dict)

        return result

    def _getitem_multi(self, index):
        data_dict = {}

        subject_idx = index // (len(self.rotations) * self.num_identities)
        rotation_idx = index % len(self.rotations)
        rotation = self.rotations[rotation_idx]
        identity_idx = index % (len(self.rotations) * self.num_identities) // len(self.rotations)

        if self.num_identities == 1:
            render_folder = osp.join(self.root, 'data', 'MultiHuman_3views', self.category, self.subject_list[subject_idx])
        else:
            render_folder = osp.join(self.root, 'data', 'MultiHuman_3views', self.category, f'{self.subject_list[subject_idx]}_{identity_idx}')

        img_path = osp.join(render_folder, 'rgb', f'{rotation:03d}.png')
        img_all_path = osp.join(render_folder, 'rgb', f'{rotation:03d}_all.png')

        img_name = self.subject_list[subject_idx]

        img_ori = load_img(img_path)
        img_pil = Image.fromarray(img_ori)
        mask_pil = Image.fromarray((np.mean(img_ori, 2) != 1))

        img_rgb = self.image_to_tensor(img_pil)
        img_mask = self.mask_to_tensor(mask_pil).float()
        data_dict['image'] = img_rgb * img_mask

        # image for hps
        if self.hps == 'pixie':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps
        elif self.hps == 'propose':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps
        
        verts, faces = obj_loader(
            osp.join(render_folder, 'scan', f'{rotation:03d}.obj')
        )

        data_dict.update(
            {
                'name': f'{self.dataset}_{self.category}_{self.subject_list[subject_idx]}_{identity_idx}_{rotation:03d}',
                'num_points': self.max_points,
                'verts': torch.as_tensor(verts).float(),
                'faces': torch.as_tensor(faces).long()
            }
        )

        return data_dict

    def _getitem_demo(self, index):
        data_dict = {}

        img_path = self.subject_list[index]
        data_dict['name'] = img_path.split('/')[-1].rsplit(".", 1)[0]

        # image loader
        data_dict['image'] = self.imagepath2tensor(img_path)

        # image for hps
        if self.hps == 'pixie':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps
        elif self.hps == 'propose':
            image_to_hps = data_dict['image']
            image_to_hps = self.image_to_hps_tensor(image_to_hps)
            data_dict['img_hps'] = image_to_hps

        data_dict['num_points'] = self.max_points

        return data_dict

    def __getitem__(self, index):
        if self.dataset in ['thuman2', 'cape']:
            return self._getitem_thuman(index)
        elif self.dataset == 'hi4d':
            return self._getitem_hi4d(index)
        elif self.dataset == 'multihuman':
            return self._getitem_multi(index)
        elif self.dataset == 'demo':
            return self._getitem_demo(index)
    
    def imagepath2tensor(self, path, channel=3):
        rgba = Image.open(path).convert('RGBA')

        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]

        return image.float()

    def load_calib(self, calib_path):
        calib_data = np.loadtxt(calib_path, dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {'calib': calib_mat}

    def load_smpl(self, data_dict):
        
        smpl_type = "smplx" if (
            'smplx_path' in data_dict.keys() and os.path.exists(data_dict['smplx_path'])
        ) else "smpl"

        return_dict = {}

        smplx_verts, _ = obj_loader(data_dict[f"{smpl_type}_path"])
        smplx_verts = torch.as_tensor(smplx_verts * self.scale).float()
        smplx_verts = projection(smplx_verts, data_dict['calib']).float()
        smplx_verts = smplx_verts * torch.tensor(np.array([[1.0, -1.0, 1.0]])).float()
        smplx_faces = torch.as_tensor(getattr(self.smplx, f"{smpl_type}_faces")).long()
        if smpl_type == "smplx":
            smplx_faces = smplx_faces[~self.smplx.smplx_eyeball_fid]
            mouth_face = torch.as_tensor(self.smplx.smplx_mouth_fid)
            smplx_faces = torch.cat([smplx_faces, mouth_face], dim=0)

        # get smpl_norms
        smplx_norms = compute_normal_batch(smplx_verts.unsqueeze(0),
                                            smplx_faces.unsqueeze(0))[0]

        return_dict.update(
            {
                'smplx_verts': smplx_verts,
                'smplx_faces': smplx_faces,
                'smplx_norms': smplx_norms
            }
        )

        return return_dict

    def load_mesh(self, data_dict):

        verts, faces = obj_loader(data_dict['mesh_path'])
        verts = torch.as_tensor(verts * self.scale).float()
        verts = projection(verts, data_dict['calib']).float()
        verts = verts * torch.tensor(np.array([[1.0, -1.0, 1.0]])).float()

        return {
            'verts': verts,
            'faces': torch.as_tensor(faces).long()
        }
    
    def load_points(self, data_dict):

        points = np.load(data_dict['points_path'])
        points = torch.as_tensor(points * self.scale).float()
        points = projection(points, data_dict['calib']).float()
        points = points * torch.tensor(np.array([[1.0, -1.0, 1.0]])).float()
        
        return {
            'points': points
        }
