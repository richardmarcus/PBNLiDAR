import numpy as np
import torch
from packaging import version as pver
from dataclasses import dataclass
# import trimesh


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")





@torch.cuda.amp.autocast(enabled=False)
def get_lidar_rays(poses, intrinsics, H, W, z_offsets, laser_offsets, N=-1, patch_size=1, scale = 0.01):
    """
    Get lidar rays.

    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = poses.device
    B = poses.shape[0]
    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float

    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])


    #z_offets is a list of 2 values
    #create 2 3d vectors from z_offsets, which represent the z_components

    
    
    z_up = torch.tensor([0, 0], dtype=torch.float32, device=z_offsets.device)
    z_down = torch.tensor([0, 0], dtype=torch.float32, device=z_offsets.device)

    z_up = torch.cat([z_up, (z_offsets[0]).unsqueeze(0)], dim=0)
    z_down = torch.cat([z_down, (z_offsets[1]).unsqueeze(0)], dim=0)

    #create n vectors by converting from lidar coordinates to world coordinates via pose_lidar
    #z_up = torch.matmul(poses[:, :3, :3], z_up.unsqueeze(-1)).squeeze(-1)
    #z_down = torch.matmul(poses[:, :3, :3], z_down.unsqueeze(-1)).squeeze(-1)

    z_offsets_up = -z_up
    z_offsets_down = -z_down


    #print(z_offsets_up)


    

    results = {}
    if N > 0:
        N = min(N, H * W)

        if isinstance(patch_size, int):
            patch_size_x, patch_size_y = patch_size, patch_size
        elif len(patch_size) == 1:
            patch_size_x, patch_size_y = patch_size[0], patch_size[0]
        else:
            patch_size_x, patch_size_y = patch_size

        if patch_size_x > 0:
            # patch-based random sampling (overlapped)
            num_patch = N // (patch_size_x * patch_size_y)
            inds_x = torch.randint(0, H - patch_size_x+1, size=[num_patch], device=device)
            inds_y = torch.randint(0, W, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]
            #print min max inds x y

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size_x, device=device),
                torch.arange(patch_size_y, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds[:, 1] = inds[:, 1] % W
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results["inds"] = inds


    #get row_indices from inds
    row_indices = inds // W
    col_indices = inds % W
    results["row_inds"] = row_indices
    results["col_inds"] = col_indices

    fov_up, fov, fov_up2, fov2 = intrinsics

    beta = -i/ W * 2 * np.pi + np.pi

    top_mask = j < (H // 2) 


    alpha = torch.zeros_like(j)
    

    laser_offsets_top = laser_offsets[j[top_mask].long()]
    laser_offsets_bottom = laser_offsets[j[~top_mask].long()]

   


    alpha_top = (fov_up -laser_offsets_top- j[top_mask] / (H//2) * fov) / 180 * np.pi
    alpha_bot = (fov_up2 -laser_offsets_bottom- (j[~top_mask]-H//2) / (H//2)* fov2) / 180 * np.pi


    alpha[top_mask] = alpha_top
    alpha[~top_mask] = alpha_bot
    '''
    # Create zero tensors for padding
    zero_top = torch.zeros(1, device=device)
    zero_middle = torch.zeros(2, device=device)
    zero_bottom = torch.zeros(1, device=device)

    # Concatenate to form combined_laser_offsets

    laser_offsets_top = laser_offsets[:H//2]
    laser_offsets_bottom = laser_offsets[H//2:]

    combined_laser_offsets = torch.cat([
        zero_top,
        laser_offsets_top,
        zero_middle,
        laser_offsets_bottom,
        zero_bottom
    ])
    
    # Apply the combined offsets
    alpha += combined_laser_offsets[j.long()]
    '''  


   
    #alpha lower is second 50% of the image

    directions = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )

    ''' 
    j_bot = j[:, W*H//2:]
    i_bot = i[:, W*H//2:]
    j = j[:, :W*H//2]
    i = i[:, :W*H//2]
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H//2* fov) / 180 * np.pi
    #alpha lower is second 50% of the image

    directions = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )
    beta = -(i_bot - W / 2) / W * 2 * np.pi
    alpha = (fov_up2 - j_bot / H//2* fov2) / 180 * np.pi
    directions_bottom = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )
    #print dir and dir bot shapes
    directions = torch.cat([directions, directions_bottom], dim=1)
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    '''
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [B, N, 3]

    #z_offsets_up = z_offsets_down

    rays_o_top = rays_o[top_mask]
    rays_o_top += z_offsets_up * scale

    rays_o_bot = rays_o[~top_mask]
    rays_o_bot += z_offsets_down * scale

    rays_shifted = torch.zeros_like(rays_o)
    rays_shifted[top_mask] = rays_o_top
    rays_shifted[~top_mask] = rays_o_bot
    rays_o = rays_shifted
 
   
    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


# def visualize_poses(poses, size=0.1):
#     # poses: [B, 4, 4]

#     axes = trimesh.creation.axis(axis_length=4)
#     box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
#     box.colors = np.array([[128, 128, 128]] * len(box.entities))
#     objects = [axes, box]

#     for pose in poses:
#         # a camera is visualized with 8 line segments.
#         pos = pose[:3, 3]
#         a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
#         b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
#         c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
#         d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

#         dir = (a + b + c + d) / 4 - pos
#         dir = dir / (np.linalg.norm(dir) + 1e-8)
#         o = pos + dir * 3

#         segs = np.array(
#             [
#                 [pos, a],
#                 [pos, b],
#                 [pos, c],
#                 [pos, d],
#                 [a, b],
#                 [b, c],
#                 [c, d],
#                 [d, a],
#                 [pos, o],
#             ]
#         )
#         segs = trimesh.load_path(segs)
#         objects.append(segs)

#     trimesh.Scene(objects).show()


@dataclass
class BaseDataset:
    pass
