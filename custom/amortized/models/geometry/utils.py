
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *


def contract_to_unisphere_custom(
    x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], unbounded: bool = False
) -> Float[Tensor, "... 3"]:
    if unbounded:
        x = scale_tensor(x, bbox, (-1, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (-1, 1))
    return x

# bug fix in https://github.com/NVlabs/eg3d/issues/67
planes =  torch.tensor(
            [
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ]
            ], dtype=torch.float32)

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=2):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(planes.to(coordinates), coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    
    # the following is from https://github.com/3DTopia/OpenLRM/blob/d4caebbea3f446904d9faafaf734e797fcc4ec89/lrm/models/rendering/synthesizer.py#L42
    output_features = output_features.permute(0, 2, 1, 3).reshape(N, M, n_planes*C)
    return output_features.contiguous()

def get_trilinear_feature(
        points: Float[Tensor, "*N Di"],
        voxel: Float[Tensor, "B Df G1 G2 G3"],
    ) -> Float[Tensor, "*N Df"]:
        b = voxel.shape[0]
        points_shape = points.shape[:-1]
        df = voxel.shape[1]
        di = points.shape[-1]
        out = F.grid_sample(
            voxel, points.view(b, 1, 1, -1, di), align_corners=False, mode="bilinear"
        )
        out = out.reshape(df, -1).T.reshape(*points_shape, df)
        return out