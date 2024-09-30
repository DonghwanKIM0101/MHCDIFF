import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAPE random masking')
    parser.add_argument(
        '--masking_ratio',
        type=float,
        default=0.2
    )
    args = parser.parse_args()
    # masking ratio -> average occlusion ratio
    # 0.05 -> 10%
    # 0.10 -> 20%
    # 0.15 -> 30%
    # 0.20 -> 40%

    image_to_tensor = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(args.masking_ratio, args.masking_ratio))
    ])

    subject_list = []
    split_txt = os.path.join('data', 'cape', 'all.txt')
    subject_list += np.loadtxt(split_txt, dtype=str).tolist()
    rotations = range(0, 360, 120)

    result_path = os.path.join('data', 'cape_masked', f'{args.masking_ratio}', 'images')
    os.makedirs(result_path, exist_ok=True)

    for subject in tqdm(subject_list):
        subject_name = subject.split("/")[1]
        render_folder = os.path.join('data', 'cape_3views', subject_name, 'render')

        for rotation in rotations:
            img_path = os.path.join(render_folder, f'{rotation:03d}.png')

            rgba = Image.open(img_path).convert('RGBA')

            mask = (cv2.imread(img_path.replace(img_path.split("/")[-2], "mask"), 0) > 1)
            img = np.asarray(rgba)[:, :, :3]
            fill_mask = ((mask & (img.sum(axis=2) == 0))).astype(np.uint8)
            image = Image.fromarray(
                cv2.inpaint(img * mask[..., None], fill_mask, 3, cv2.INPAINT_TELEA)
            )
            mask = Image.fromarray(mask)

            image = image_to_tensor(image)
            
            torchvision.utils.save_image(image, os.path.join(result_path, f'cape_{subject_name}_{rotation:03d}.png'))
