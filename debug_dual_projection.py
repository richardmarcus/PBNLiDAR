
import numpy as np
import cv2

FOV_LIDAR = [1.9647572, 11.0334425, -8.979475, 16.52717]
Z_OFFSETS = [-0.20287499, -0.12243641]


laser_offsets = [
    0.0101472, 0.02935141, -0.04524597, 0.04477938, -0.00623795, 0.04855699,
    -0.02581356, -0.00632023, 0.00133613, 0.05607248, 0.00494516, 0.00062785,
    0.03141189, 0.02682017, 0.01036519, 0.02891498, -0.01124913, 0.04208804,
    -0.0218643, 0.00743873, -0.01018788, -0.01669445, 0.00017374, 0.0048293,
    0.03166919, 0.03558188, 0.01552001, -0.03950449, 0.00887087, 0.04522041,
    -0.04557779, 0.01275884, 0.02858396, 0.06113308, 0.03508026, -0.07183428,
    -0.10038704, 0.02749107, 0.0291795, -0.03833354, -0.07382096, -0.14437623,
    -0.09460489, -0.0584761, 0.01881664, -0.02696179, -0.02052307, -0.15732896,
    -0.03719316, -0.00687183, 0.07373429, 0.03398049, 0.04429062, -0.05352834,
    -0.07988049, -0.02726229, -0.00934669, 0.09552395, 0.0850026, -0.00946006,
    -0.05684165, 0.0798225, 0.10324192, 0.08222152
]



pcd_path = "./data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000200.bin"

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
    

            
    if mask is None:
        mask = (r >= 0) & (r < lidar_H)


    
    r = r[mask]

    return r

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

    alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + (fov_down+laser_offsets) / 180 * np.pi
    r = (lidar_H - alpha / (fov / 180 * np.pi / lidar_H))+0.5



    beta = np.pi - np.arctan2(y, x)
    c = (beta / (2 * np.pi / lidar_W))+0.5

    #stack xyz
    local_points = np.stack([x, y, z], axis=1)


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


def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, z_offsets, laser_offsets, max_depth=80.0, use_offsets=False
):
    points = local_points_with_intensities[:, :3]

    mask = np.ones(points.shape[0], dtype=bool)
    mask = np.linalg.norm(points, axis=1) > 2.5
    #also mask distance greater than 80m and lower than 2m
    mask = mask & (np.linalg.norm(points, axis=1) < 80) &  (points[:, 2] > -2)
    local_points_with_intensities = local_points_with_intensities[mask]

    if use_offsets:
        y_r, y_r2 = compare_lidar_to_pano_with_intensities(
            local_points_with_intensities=local_points_with_intensities,
            lidar_H=lidar_H,
            lidar_W=lidar_W,
            lidar_K=intrinsics,
            z_offsets=z_offsets,
            max_depth=max_depth
        )
        ring_ids = np.concatenate((y_r, y_r2+lidar_H//2))

        ring_ids = ring_ids.astype(float) /lidar_H *64

        
        spread_laser_offsets = laser_offsets[ring_ids.astype(np.int32)]

        if lidar_H < 64:
            #if lidar_H is less than 64, we use dummy offsets
            spread_laser_offsets = np.zeros((local_points_with_intensities.shape[0],), dtype=np.float32)

        print(local_points_with_intensities.shape, spread_laser_offsets.shape)
        assert local_points_with_intensities.shape[0] == spread_laser_offsets.shape[0]

    else:
        spread_laser_offsets = np.zeros((local_points_with_intensities.shape[0],), dtype=np.float32)

    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        z_offsets=z_offsets,
        max_depth=max_depth,
        laser_offsets= spread_laser_offsets
    )


    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


#main function to test the conversion
if __name__ == "__main__":
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)

    #H should be 64, > 64 uses dummy offsets, <64 uses 0 for offsets
    range_view =  LiDAR_2_Pano_KITTI(points,
                                    lidar_H=64,
                                    lidar_W=1024,
                                    intrinsics=FOV_LIDAR,
                                    z_offsets=Z_OFFSETS,
                                    laser_offsets=np.array(laser_offsets, dtype=np.float32),
                                    max_depth=80.0,
                                    use_offsets=False)
    
    #imshow
    cv2.imshow("range_view", range_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()