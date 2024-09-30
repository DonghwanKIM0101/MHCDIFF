import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.ops import masks_to_boxes


def process_image(img_ori, mask, input_res=512):
    
    in_height, in_width, _ = img_ori.shape
    M = aug_matrix(in_width, in_height, input_res * 2, input_res * 2)

    # from rectangle to square
    img_for_crop = cv2.warpAffine(
        img_ori, M[0:2, :], (input_res * 2, input_res * 2), flags=cv2.INTER_CUBIC
    )

    mask = np.expand_dims(mask, 2)
    mask_for_crop = cv2.warpAffine(
        mask.astype(np.uint8), M[0:2, :], (input_res * 2, input_res * 2), flags=cv2.INTER_CUBIC
    )
    img_for_crop = img_for_crop * np.expand_dims(mask_for_crop, 2)

    bbox = masks_to_boxes(torch.tensor(mask_for_crop).unsqueeze(0))[0]
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

    scale = max(height, width) / 180

    img_np = crop(img_for_crop, center, scale, (input_res, input_res))
    
    return img_np

def load_img(img_file):

    if img_file.endswith("exr"):
        img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    else :
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    # considering non 8-bit image
    if img.dtype != np.uint8 :
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img

def aug_matrix(w1, h1, w2, h2):
    dx = (w2 - w1) / 2.0
    dy = (h2 - h1) / 2.0

    matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

    scale = np.min([float(w2) / w1, float(h2) / h1])

    M = get_affine_matrix(center=(w2 / 2.0, h2 / 2.0), translate=(0, 0), scale=scale)

    M = np.array(M + [0., 0., 1.]).reshape(3, 3)
    M = M.dot(matrix_trans)

    return M

def get_affine_matrix(center, translate, scale):
    cx, cy = center
    tx, ty = translate

    M = [1, 0, 0, 0, 1, 0]
    M = [x * scale for x in M]

    # Apply translation and of center translation: RSS * C^-1
    M[2] += M[0] * (-cx) + M[1] * (-cy)
    M[5] += M[3] * (-cx) + M[4] * (-cy)

    # Apply center translation: T * C * RSS * C^-1
    M[2] += cx + tx
    M[5] += cy + ty
    return M

def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    return t

def transform(pt, center, scale, res, invert=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.around(new_pt[:2]).astype(np.int16)

def crop(img, center, scale, res):
    """Crop image according to the supplied bounding box."""

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))

    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    if len(img.shape) == 2:
        new_img = np.array(Image.fromarray(new_img).resize(res))
    else:
        new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res))

    return new_img
