from matplotlib import pyplot as plt
import numpy as np
import torch
import math

def lidar_to_pano_with_intensities_half(local_points_with_intensities, lidar_H, lidar_W, fov, fov_down, max_depth=80, z_off=0, bot = False, double = True, ring=True, mask=None, laser_offsets=0):


    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    # Extract coordinates
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2] 
    
    if double:
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)

    z+= z_off

    dists = np.sqrt(x**2 + y**2 + z**2)
    #valid_mask = dists < max_depth
    #local_point_intensities = local_point_intensities[valid_mask]
    #dists = dists[valid_mask]
    #x = x[valid_mask]
    #y = y[valid_mask]
    #z = z[valid_mask]

    alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + (fov_down+laser_offsets) / 180 * np.pi
    r = (lidar_H - alpha / (fov / 180 * np.pi / lidar_H))+0.5



    beta = np.pi - np.arctan2(y, x)
    c = (beta / (2 * np.pi / lidar_W))+0.5

    #stack xyz
    local_points = np.stack([x, y, z], axis=1)

    if ring:
        y_r = calculate_ring_ids(local_points, lidar_H*2)
        if bot:
            y_r-=lidar_H

    if ring:
        r = y_r
    mask = (r >= 0) & (r < lidar_H) 
    r = r[mask]
    c = c[mask]
    dists = dists[mask]
    local_point_intensities = local_point_intensities[mask]

        # Initialize pano and intensity maps
    pano = np.zeros((lidar_H, lidar_W), dtype=np.float64)
    intensities = np.zeros((lidar_H, lidar_W), dtype=np.float32)


    r = r.astype(np.int32)
    c = c.astype(np.int32)%lidar_W

    for y, x, dist, intensity in zip(r, c, dists, local_point_intensities):
        
     
        if pano[y,x] == 0:
            pano[y,x] = dist
            intensities[y,x] = intensity
        elif pano[y,x] > dist:
            pano[y,x] = dist
            intensities[y,x] = intensity

    return pano, intensities


def lidar_to_pano_with_intensities_half_y_coord(local_points_with_intensities, lidar_H, lidar_W, fov, fov_down, max_depth=80, z_off=0, bot = False, double = True, ring=True, mask=None, laser_offsets=None, rids = None):

    if bot & (mask is not None):
        mask = ~mask

    local_points = local_points_with_intensities[:, :3]
    # Extract coordinates
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2] 
    if double:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
    if laser_offsets is None:
        laser_offsets = 0

  
    
    alpha = np.arctan2(z+z_off, np.sqrt(x**2 + y**2)) + (fov_down +laser_offsets) / 180 * np.pi 
    r = (lidar_H - alpha / (fov / 180 * np.pi / lidar_H))+0.5

    local_points = np.stack([x, y, z], axis=1)
    
    #mask = (r >= 0) & (r < lidar_H) 
    if ring:
        y_r = calculate_ring_ids(local_points, lidar_H*2)
        #stack xyz
        r = y_r
        if bot:
            y_r-=lidar_H
            
    if mask is None:
        #print("careful")
        mask = (r >= 0) & (r < lidar_H)
    #    mask = (y_r < lidar_H) & (y_r >= 0)

    
    r = r[mask]

    #print(rids, mask)

    
    return r

def lidar_to_pano_with_intensities_half_y_coord_torch(local_points_with_intensities, lidar_H, lidar_W, fov, fov_down, max_depth=80, z_off=0, bot=False, double=True, ring=True, mask=None, laser_offsets=None):

    points = local_points_with_intensities[:, :3]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    #convert to float32
    x = x.float()
    y = y.float()
    z = z.float()

    alpha = torch.atan2(z+z_off, torch.sqrt(x**2 + y**2)) + (fov_down + laser_offsets) / 180.0 * math.pi
    r = lidar_H - alpha / ((fov / 180.0) * math.pi / lidar_H)
    
    return r[mask]


def get_quadrant(point):
        res = 0
        x = point[0]
        y = point[1]
        if x > 0 and y >= 0:
            res = 1
        elif x <= 0 and y > 0:
            res = 2
        elif x < 0 and y <= 0:
            res = 3
        elif x >= 0 and y < 0:
            res = 4
        return res

def calculate_ring_ids(scan_points, height=256):

    #print("dont use this for now")
    #print(5/0)
    #exit()
    #print(scan_points.shape)
    num_of_points = scan_points.shape[0]
    velodyne_rings_count = 64
    previous_quadrant = 0
    ring = 0
    ring_ids = np.zeros(num_of_points, dtype=np.int32)
    for num in range(num_of_points-1, -1, -1):
        quadrant = get_quadrant(scan_points[num])
        #print(quadrant)
        if quadrant == 4 and previous_quadrant == 1 and ring < velodyne_rings_count-1:
            ring += 1

        ring_ids[num] = height - int(height/64)*ring - height//64
        #print(ring, num)

        previous_quadrant = quadrant


    return ring_ids

def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    z_offsets,
    max_depth=80,
    ring=False,
    mask=None,
    laser_offsets=0

):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Unpack lidar intrinsics

    fov_up, fov, fov_up2, fov2 = lidar_K
    fov_down = fov - fov_up
    fov_down2 = fov2 - fov_up2

    pano, intensities = lidar_to_pano_with_intensities_half(local_points_with_intensities, lidar_H//2, lidar_W, fov, fov_down, max_depth, z_offsets[0], ring=ring, mask=mask, laser_offsets=laser_offsets)

    pano2, intensities2 = lidar_to_pano_with_intensities_half(local_points_with_intensities, lidar_H//2, lidar_W, fov2, fov_down2, max_depth, z_offsets[1], bot=True, ring=ring, mask=mask, laser_offsets=laser_offsets)

    #stack
    pano = np.concatenate([pano, pano2], 0)
    intensities = np.concatenate([intensities, intensities2], 0)
    return pano, intensities


def compare_lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    z_offsets,
    max_depth=80,
    ring=False,
    mask=None,
    laser_offsets=None,
    rids =  None

):

    fov_up, fov, fov_up2, fov2 = lidar_K
    fov_down = fov - fov_up
    fov_down2 = fov2 - fov_up2

    ys= lidar_to_pano_with_intensities_half_y_coord(local_points_with_intensities, lidar_H//2, lidar_W, fov, fov_down, max_depth, z_offsets[0], ring=ring, mask=mask, laser_offsets=laser_offsets,rids=rids)

    ys2= lidar_to_pano_with_intensities_half_y_coord(local_points_with_intensities, lidar_H//2, lidar_W, fov2, fov_down2, max_depth, z_offsets[1], bot=True, ring=ring , mask=mask, laser_offsets=laser_offsets, rids = rids)


   
    return ys,ys2


def compare_lidar_to_pano_with_intensities_torch(
    local_points_with_intensities,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    z_offsets,
    max_depth=80,
    ring=True,
    mask=None,
    laser_offsets=None

):

    fov_up, fov, fov_up2, fov2 = lidar_K
    fov_down = fov - fov_up
    fov_down2 = fov2 - fov_up2

    ys= lidar_to_pano_with_intensities_half_y_coord_torch(local_points_with_intensities, lidar_H//2, lidar_W, fov, fov_down, max_depth, z_offsets[0], ring=ring, mask=mask, laser_offsets=laser_offsets)

    ys2= lidar_to_pano_with_intensities_half_y_coord_torch(local_points_with_intensities, lidar_H//2, lidar_W, fov2, fov_down2, max_depth, z_offsets[1], bot=True, ring=ring, mask=~mask, laser_offsets=laser_offsets)

   
    return ys,ys2


def lidar_to_pano(
    local_points: np.ndarray, lidar_H: int, lidar_W: int, lidar_K: int, max_dpeth=80
):
    """
    Convert lidar frame to pano frame. Lidar points are in local coordinates.

    Args:
        local_points: (N, 3), float32, in lidar frame.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
    """

    # (N, 3) -> (N, 4), filled with zeros.
    local_points_with_intensities = np.concatenate(
        [local_points, np.zeros((local_points.shape[0], 1))], axis=1
    )
    pano, _ = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=lidar_K,
        max_dpeth=max_dpeth,
    )
    return pano


def pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K, z_offsets, laser_offsets=0):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """

    #convert lidar_k and z_offsets from torch to numpy
    
    if torch.is_tensor(lidar_K):
        lidar_K = lidar_K.cpu().detach().numpy()
        z_offsets = z_offsets.cpu().detach().numpy()
        laser_offsets = laser_offsets.cpu().detach().numpy()

    fov_up, fov, fov_up2, fov2 = lidar_K

    H, W = pano.shape
    H = H//2
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )


 

    beta = -(i ) / (W) * 2 * np.pi + np.pi


    alpha = (fov_up - laser_offsets[j.astype(int)] - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    alpha = (fov_up2 - laser_offsets[j.astype(int)+32] - j / H * fov2) / 180 * np.pi
    dirs_bot = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
  
    #stack
    dirs = np.concatenate([dirs, dirs_bot], 0)

    '''
    dirs_flat = dirs.reshape(-1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for direction in dirs_flat:
        direction = direction * 80
        ax.plot([0,   direction[0]],
                [0,  direction[1]],
                [0, direction[2]], color='b', linewidth=0.5)
        
    plt.show()
    '''
    origin_up = [0, 0, -z_offsets[0]]
    origin_bot = [0, 0, -z_offsets[1]]
    local_points = np.zeros((H*2, W, 3))
    #TODO_C check offsets
    local_points[:H, :, :] = origin_up + dirs[:H, :, :] * pano.reshape(H*2, W, 1)[:H, :, :]
    local_points[H:, :, :] = origin_bot + dirs[H:, :, :] * pano.reshape(H*2, W, 1)[H:, :, :]

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H*2, W, 1)], axis=2
    )

    
   
    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]
    #flatten
    #local_points_with_intensities = local_points_with_intensities.reshape(-1, 4)
    #print(local_points_with_intensities.shape)

    return local_points_with_intensities


def pano_to_lidar(pano, lidar_K, z_offsets, laser_offsets):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
        z_offsets = z_offsets,
        laser_offsets=laser_offsets
    )
    return local_points_with_intensities[:, :3]