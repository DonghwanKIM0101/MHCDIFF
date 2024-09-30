import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform
)

def get_camera(device, scale=100.0):
    cam_pos = (0, 0, scale)
    R,T = look_at_view_transform(
        eye=[cam_pos],
        at=((0, 0, 0), ),
        up=((0, 1, 0), ),
    )

    camera = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=scale,
        zfar=-scale,
        max_y=scale,
        min_y=-scale,
        max_x=scale,
        min_x=-scale,
        scale_xyz=(scale * np.ones(3), ),
    )

    return camera
