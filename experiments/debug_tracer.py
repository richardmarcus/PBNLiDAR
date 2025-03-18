import torch
import numpy as np
import cv2
import os
from data.base_dataset import get_lidar_rays
from utils.convert import  lidar_to_pano_with_intensities, calculate_ring_ids, pano_to_lidar_with_intensities, compare_lidar_to_pano_with_intensities
from mpl_toolkits.mplot3d import Axes3D
from data.preprocess.kitti360_loader import KITTI360Loader
from matplotlib import pyplot as plt
from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from concurrent.futures import ThreadPoolExecutor

from model.runner import interpolate_poses


device = "cuda" if torch.cuda.is_available() else "cpu"

path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

path_mc = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_deskewed/"

cam_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/"

kitti_360_root = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360"
# Load KITTI-360 dataset.
k3 = KITTI360Loader(kitti_360_root)
sequence_name = "2013_05_28_drive_0000"

skip = 1
plt.figure(figsize=(10, 10))
num_seq = 7
fid = 1545#1780

num_merge = num_seq*2

interpolate_jump = 1

print("core frame ", fid)

for first_frame in range(fid-num_seq, fid+num_seq):
    print(first_frame)

    frame_ids = list(range(first_frame, first_frame+num_merge))
    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)
    pose_before = k3.load_lidars(sequence_name, [first_frame-1-interpolate_jump])[0]
    pose_after = k3.load_lidars(sequence_name, [first_frame+num_merge+interpolate_jump])[0]
    cam_to_world_dict, p_rect_00, cam_00_to_velo, r_rect_00 = k3.load_cameras(sequence_name, frame_ids)

    mc_poses_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/icp_poses.txt"
    #format is 1.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00 0.000000000000000000e+00
    mc_poses = []

    print("reading mc poses")
    with open(mc_poses_path, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            line = [float(i) for i in line]
            pose = np.array(line).reshape(3, 4)
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
            mc_poses.append(pose)


    ray_res_x = 1024
    ray_res_y = 64

    #K = [1.9050592, 10.385118,  -8.774327,  15.891012]


    K = [2.02984126984, 11.0317460317, -8.799812, 16.541]


    z_offsets = [-0.202, -0.121]

    laser_offsets = [ 0.0101472,   0.02935141, -0.04524597,  0.04477938, -0.00623795,  0.04855699, 
    -0.02581356, -0.00632023,  0.00133613,  0.05607248,  0.00494516,  0.00062785,
    0.03141189,  0.02682017,  0.01036519,  0.02891498, -0.01124913,  0.04208804,
    -0.0218643,   0.00743873, -0.01018788, -0.01669445,  0.00017374,  0.0048293,
    0.03166919,  0.03558188,  0.01552001, -0.03950449,  0.00887087,  0.04522041,
    -0.04557779,  0.01275884,  0.02858396,  0.06113308,  0.03508026, -0.07183428,
    -0.10038704,  0.02749107,  0.0291795,  -0.03833354, -0.07382096, -0.14437623,
    -0.09460489, -0.0584761,   0.01881664, -0.02696179, -0.02052307, -0.15732896,
    -0.03719316, -0.00687183,  0.07373429,  0.03398049,  0.04429062, -0.05352834,
    -0.07988049, -0.02726229, -0.00934669,  0.09552395,  0.0850026,  -0.00946006,
    -0.05684165,  0.0798225,   0.10324192,  0.08222152]

    #set to 0
    if ray_res_y != 64 or False:
        laser_offsets = [0 for i in range(ray_res_y)]



    K = [ 1.9647572, 11.0334425, -8.979475,  16.52717 ]
    z_offsets = [-0.20287499, -0.12243641 ]

    laser_offsets = np.array(laser_offsets)

    #plot x and y positions of points
    #increase dpi
    #plt.figure(dpi=1000)
    #figsize
    #plt.figure(figsize=(80, 80))

    plt_img = np.zeros((4000,4000,3))
    #uint3
    plt_img = plt_img.astype(np.uint8)
    files = os.listdir(path)
    files.sort()
    for p in range(0,1):
        use_icp = p == 1

        big_pcd= []
        big_intensities = []
        base_pose = lidar2world[0]
        base_mc_pose = mc_poses[first_frame]
        for file in files:
            if int(file.split(".")[0]) < first_frame:
                continue
            bin_pcd = np.fromfile(os.path.join(path, file), dtype=np.float32).reshape(-1,4)
            intensity = bin_pcd[:,3]
            bin_pcd = bin_pcd[:, :3]
            if use_icp or True:
                mc_pcd = np.fromfile(os.path.join(path_mc, file), dtype=np.float32).reshape(-1,3)
                bin_pcd = mc_pcd


            #downsample 
            bin_pcd = bin_pcd[::skip]
            intensity = intensity[::skip]
            #remove points in a 1m radius
            #init empty mask
            mask = np.ones(bin_pcd.shape[0], dtype=bool)
            mask = np.linalg.norm(bin_pcd, axis=1) > 2.6
            #also mask distance greater than 80m and lower than 2m
            mask = mask & (np.linalg.norm(bin_pcd, axis=1) < 80) &  (bin_pcd[:, 2] > -2)
            bin_pcd = bin_pcd[mask]
            intensity = intensity[mask]
            big_intensities.append(intensity)


            if num_merge > 1:
                        #add homogenous coordinates
                bin_pcd = np.concatenate([bin_pcd, np.ones((bin_pcd.shape[0], 1))], axis=1)

                pose = lidar2world[len(big_pcd)]
                if use_icp:
                    pose = mc_poses[len(big_pcd)+first_frame]
                    base_pose = base_mc_pose
                #bring into base pose coordinates
                pose = np.linalg.inv(base_pose) @ pose
                bin_pcd = pose @ bin_pcd.T
                #remove homogenous coordinates
                bin_pcd = bin_pcd.T[:, :3]
            big_pcd.append(bin_pcd)
            print("add frameid" , file)

            if len(big_pcd) == num_merge:
                break

        big_pcd = np.concatenate(big_pcd)
        big_intensities = np.concatenate(big_intensities)
        big_pcd = np.concatenate([big_pcd, big_intensities[:, None]], axis=1)

        if False:
            #color depending on p 
            if p == 0:
                cmap = 'Reds'
                color = (1, 0, 0)
            else:
                cmap = 'Blues'
                color = (0, 0, 1)

            print(cmap, "icp", use_icp)

            #normalize x and y between 0 and 1
            big_pcd[:,0] = (big_pcd[:,0] - np.min(big_pcd[:,0])) / (np.max(big_pcd[:,0]) - np.min(big_pcd[:,0]))
            big_pcd[:,1] = (big_pcd[:,1] - np.min(big_pcd[:,1])) / (np.max(big_pcd[:,1]) - np.min(big_pcd[:,1]))

            #mapped_col = plt.get_cmap(cmap)(big_pcd[:,3])
            #remove alpha
            #mapped_col = mapped_col[:, :3]
            #to uint8
            #print(mapped_col)

            mapped_col = big_pcd[:,3][:, None] * np.array(color)[None, :]
            #uint8
            mapped_col = (mapped_col * 255).astype(np.uint8)

            plt_img[(big_pcd[:,0]*(plt_img.shape[0]-1)).astype(int), (big_pcd[:,1]*(plt_img.shape[1]-1)).astype(int)] += mapped_col


        #plt.gca().set_facecolor('white')
        #plt.scatter(big_pcd[:,0], big_pcd[:,1], marker='.', s=1, linewidths=0, c=1-big_pcd[:,3], cmap=cmap)

        #make sure x and y are equal
        #plt.gca().set_aspect('equal', adjustable='box')
        
        #c=big_pcd[:,2], cmap='viridis')


    plt_img = cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("debug_proj/xy.png", plt_img)
    print("saved to debug_proj/xy.png")

    RT = True
    PROJECT = True
    print(big_pcd.shape)
    render_frame = first_frame + num_seq


    for file in files:
        if int(file.split(".")[0]) < render_frame:
            continue

        if int(file.split(".")[0]) >= render_frame+ num_seq:
            exit()


        if PROJECT:
            bin_pcd = big_pcd# np.fromfile(os.path.join(path, file), dtype=np.float32).reshape(-1,4)
            #print max and min distance
            intensity = bin_pcd[:,3]
   


            points = bin_pcd[:, :3]
                    
            #add homogenous coordinates
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            pose = lidar2world[int(file.split(".")[0])-first_frame]
            pose = np.linalg.inv(pose) @ base_pose
            points = pose @ points.T
            #remove homogenous coordinates
            points = points.T[:, :3]

            intensity = intensity[points[:,0] > 0]
            points = points[points[:,0] > 0]

            # P_rect_00: Projektionsmatrix
            P_rect_00 = np.array([[552.554261, 0.000000, 682.049453, 0.000000],
                                [0.000000, 552.554261, 238.769549, 0.000000],
                                [0.000000, 0.000000, 1.000000, 0.000000],
                                [0.000000, 0.000000, 0.000000, 1.000000]])


            # Transformation von Kamera zu Velodyne
            # calib_path = "calibration/calib_cam_to_velo.txt"
            calib_cam_to_velodyne = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                                [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                                [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                                [0.000000, 0.000000, 0.000000, 1.000000]])
            # Invertieren
            calib_velodyne_to_cam = np.linalg.inv(calib_cam_to_velodyne)
            # in homogene Koordinaten umwandeln (X, Y, Z, 1)
            points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))

            # Transformation in Kamerakoordinaten
            transformed_points = np.dot(points_homogeneous, calib_velodyne_to_cam.T)

            # 3D Punkte in 2D projizieren
            image_points = np.dot(transformed_points, P_rect_00.T)
            # Dehomogenisierung
            image_points[:, 0] /= image_points[:, 2]
            image_points[:, 1] /= image_points[:, 2]
        
            #write points into camera image
            cam_img = cv2.imread(os.path.join(cam_path, file.split(".")[0] + ".png"))

            blackimg = np.zeros((376, 1408, 3), dtype=float)
            fillimg = np.zeros((376, 1408, 3), dtype=float)
            mask = (image_points[:, 0] >= 0) & (image_points[:, 0] < 1408) & (image_points[:, 1] >= 0) & (image_points[:, 1] < 376)
            image_points = image_points[mask]
            intensity = intensity[mask]
            u = image_points[:, 0].astype(np.uint16)
            v = image_points[:, 1].astype(np.uint16)
            z = image_points[:, 2]
            col = (intensity* 255)
            rgb_col = np.stack([col, col, col], axis=-1)
            #set blue to 0
            rgb_col[:, 0] = 100
            rgb_col[:, 1] = z

            u = u.astype(int)
            v = v.astype(int)

            if True:
                # Vectorize the process of finding the smallest z for each (u, v) pair
                unique_uv, unique_indices = np.unique(np.stack([u, v], axis=-1), axis=0, return_inverse=True)
                
                smallest_z = np.full(len(unique_uv), np.inf)

                # Use np.minimum.at to find the smallest z for each unique (u, v) pair
                np.minimum.at(smallest_z, unique_indices, z)

                # Create a boolean mask indicating which points have the smallest z for their (u, v) pair
                mask = smallest_z[unique_indices] == z

                # Filter the u, v, col arrays to keep only the desired points
                u = u[mask]
                v = v[mask]
                z = z[mask]

                blackimg[v,u]= rgb_col[mask]

            else:
                blackimg[v,u]= rgb_col

    
            if True:
                #for each pixel: color is average of surrounding pixels within depth threshold
                def process_pixel(i, j):
                    surrounding = blackimg[i-1:i+2, j-1:j+2]
                    if np.any(surrounding != 0):
                        #compare z values
                        surrounding_z = blackimg[i-1:i+2, j-1:j+2, 1]
                        closest_z = np.min(surrounding_z[surrounding_z != 0])
                        
                        close_surrounding = surrounding[np.abs(surrounding_z - closest_z) <= 0.1]#.001]

                        if len(close_surrounding) > 0:
                            return i, j, np.mean(close_surrounding, axis=0).astype(np.uint8)
                    return None, None, None

                # Prepare a list of pixel coordinates to process
                pixels = [(i, j) for i in range(1, 375) for j in range(1, 1407)]

                # Use a ThreadPoolExecutor to parallelize the processing
                with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers as needed
                    results = executor.map(lambda p: process_pixel(*p), pixels)

                # Apply the results to fillimg
                for i, j, color in results:
                    if i is not None and j is not None and color is not None:
                        fillimg[i, j] = color

        
                depth_diff = np.abs(fillimg - blackimg)[:, :, 1]
                #if depth_diff > 0.1, set blackimg to fillimg
                mask = (depth_diff > 0.5) | (blackimg[:, :, 1] == 0)
                blackimg[mask] = fillimg[mask]  
                #blackimg = fillimg
            #set green to 0
            #blackimg[:, :, 0] = 0
            #blackimg[:, :, 2] = 0
            blackimg[:, :, 1] = 0
            #blackimg[:, :, 1] = 1 - blackimg[:, :, 1] *3
            #multiply by 255
         
            blackimg = blackimg.astype(np.uint8)
            cam_img =cv2.addWeighted(cam_img, 0.3, blackimg, 0.7, 0)
            
                
            cv2.imwrite("debug_proj/"+file.split(".")[0] + "_cam.png", cam_img)
            print("saved to debug_proj/"+file.split(".")[0] + "_cam.png")
            break

            ring_ids = calculate_ring_ids(bin_pcd, 64)
            spread_laser_offsets = np.array(laser_offsets)[ring_ids]


            pano, intensities = lidar_to_pano_with_intensities(bin_pcd, ray_res_y, ray_res_x, K, z_offsets, ring=False, laser_offsets=spread_laser_offsets)
            rgb_empty = np.zeros((pano.shape[0], pano.shape[1], 3), dtype=np.uint8)


            #if True:
            if False:
                repro = pano_to_lidar_with_intensities(pano, intensities, K, z_offsets, laser_offsets)

                #chamfer distance between repro and bin_pcd
                repro_without_intensities = repro[:, :3]
                bin_pcd_without_intensities = bin_pcd[:, :3]

                print(repro_without_intensities.shape, bin_pcd_without_intensities.shape)

                #chamfer distance
                repro_torch = torch.tensor(repro_without_intensities).to(device)
                bin_pcd_torch = torch.tensor(bin_pcd_without_intensities).to(device)
                #convert to float32
                repro_torch = repro_torch.float()
                bin_pcd_torch = bin_pcd_torch.float()
                chamfer = chamfer_3DDist()
                dist1, dist2, _, _ = chamfer(repro_torch.unsqueeze(0), bin_pcd_torch.unsqueeze(0))
                chamfer_loss = dist1.mean() + dist2.mean() 
                print("chamfer loss", chamfer_loss.item()*0.5)
            


            
                #ys_ring, ys_ring2 = compare_lidar_to_pano_with_intensities(repro, ray_res_y, ray_res_x, K, z_offsets, spread_laser_offsets)
                #ring_ids = np.concatenate((ys_ring, ys_ring2+32))

                ring_ids = calculate_ring_ids(repro, 64)

                spread_laser_offsets = np.array(laser_offsets)[ring_ids]
                pano_re, intensities_re = lidar_to_pano_with_intensities(repro, ray_res_y, ray_res_x, K, z_offsets, ring=False)

                
                #write pano_re and pano with cv2
                pano_img = cv2.cvtColor(pano.astype(np.float32), cv2.COLOR_GRAY2BGR)
                #make it white where pano is not 0
                pano_img[pano != 0] = [255, 255, 255]
                cv2.imwrite("debug_proj/pano.png", pano_img)
                pano_re_img = cv2.cvtColor(pano_re.astype(np.float32), cv2.COLOR_GRAY2BGR)
                pano_re_img[pano_re != 0] = [255, 255, 255]
                cv2.imwrite("debug_proj/pano_re.png", pano_re_img)

                img_diff = np.abs(pano_img - pano_re_img)

            
                
            
                cv2.imwrite("debug_proj/diff.png", img_diff)
            

                repro2 = pano_to_lidar_with_intensities(pano_re, intensities_re, K, z_offsets, laser_offsets)
                


                #ring_ids = calculate_ring_ids(repro2, 64)
                #spread_laser_offsets = np.array(laser_offsets)[ring_ids]
                #pano_re2, intensities_re2 = lidar_to_pano_with_intensities(repro2, ray_res_y, ray_res_x, K, z_offsets, ring=False)
                #pano_re2_img = cv2.cvtColor(pano_re2, cv2.COLOR_GRAY2BGR)
                #pano_re2_img[pano_re2 != 0] = [255, 255, 255]
                #cv2.imwrite("debug_proj/pano_re2.png", pano_re2_img)
                


                #vis repro and repro2 with open3d
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(bin_pcd[:, :3])
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(repro_without_intensities)

                #color blue and red
                pcd.paint_uniform_color([0, 0, 1])
                pcd2.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([pcd, pcd2])

                repro2_without_intensities = repro2[:, :3]
                diff = np.linalg.norm(repro_without_intensities - repro2_without_intensities, axis=1)

                print("diff", np.mean(diff), np.median(diff), np.max(diff), np.min(diff))
                exit()

                intensities = (intensities * 255).astype(np.uint8)
                #set rgb to purple where pano is not 0
                rgb_empty[pano == 0] = [255, 0, 255]
                #set rgb to intensity where pano is not 0
                rgb_empty[pano != 0] = np.stack([intensities[pano != 0]] * 3, axis=-1).astype(np.uint8)
            

                cv2.imwrite("debug_proj/proj.png", rgb_empty)
                exit()

        #poses_lidar is 1, identity matrix, shape [1, 4, 4]
        #intrinsics is K, shape [4]
        #H_lidar is ray_res_y, W_lidar is ray_res_x
        exit()
        print(z_offsets, file.split(".")[0], render_frame)
        z_offsets = torch.tensor(z_offsets).to(device)
        laser_offsets = torch.tensor(laser_offsets).to(device)


        pose = lidar2world[render_frame-first_frame]

        pose = torch.tensor(pose).to(device)

        if render_frame == first_frame:
            prev_pose = pose_before
        else:
            prev_pose = lidar2world[render_frame-first_frame-1]

        if render_frame == first_frame+num_merge-1:
            next_pose = pose_after
            print(render_frame, first_frame+ num_merge)
        else:
            next_pose = lidar2world[render_frame-first_frame+1]

        prev_pose = torch.tensor(prev_pose).to(device)
        next_pose = torch.tensor(next_pose).to(device)

        #laseroffsets to float
        laser_offsets = laser_offsets.float()

        pose = pose.unsqueeze(0).float()
        prev_pose = prev_pose.unsqueeze(0).float()
        next_pose = next_pose.unsqueeze(0).float()

        


        if RT:
            rays_lidar = get_lidar_rays(
                pose,
                K,
                ray_res_y,
                ray_res_x,
                z_offsets,
                laser_offsets,
                -1,
                1,
                1
            )
            import matplotlib.pyplot as plt


            prev_trans = prev_pose[:, :3, 3]
            trans = pose[:, :3, 3]
            next_trans = next_pose[:, :3, 3]

      

            #to cpu
            prev_trans = prev_trans.cpu().numpy()[0]
            trans = trans.cpu().numpy()[0]
            next_trans = next_trans.cpu().numpy()[0]

            prev_forward = prev_pose[:, :3, 1]
            forward = pose[:, :3, 1]
            next_forward = next_pose[:, :3, 1]

            #to cpu
            prev_forward = prev_forward.cpu().numpy()[0]
            forward = forward.cpu().numpy()[0]
            next_forward = next_forward.cpu().numpy()[0]

            #remove z component and normalize
            prev_forward = prev_forward[:2] / np.linalg.norm(prev_forward[:2])
            forward = forward[:2] / np.linalg.norm(forward[:2])
            next_forward = next_forward[:2] / np.linalg.norm(next_forward[:2])



            interpolated_poses = interpolate_poses(pose, prev_pose, next_pose, 1024)
            #unsqueeze


            #first batch only relevant
            rays_o = rays_lidar["rays_o"][0]
            rays_d = rays_lidar["rays_d"][0]
            row_inds = rays_lidar["row_inds"][0]
            col_inds = rays_lidar["col_inds"][0]

            interpolated_poses = interpolated_poses[0][col_inds]
            

            forward_vectors = interpolated_poses[:, :3, 3] - rays_o
            rays_o = rays_o + forward_vectors
            motion_rotation = interpolated_poses[:, :3, :3]

            interpolated_forwards = interpolated_poses[:, :3, 1]


            base_rotation = pose[:, :3, :3]
            inv_rotation = torch.inverse(base_rotation)
            relative_rotation = torch.matmul(inv_rotation, motion_rotation)



            rays_d =torch.matmul(relative_rotation, rays_d.unsqueeze(-1)).squeeze(-1)


            #only take row 32
            rays_o = rays_o[row_inds == 32]
            rays_d = rays_d[row_inds == 32]
            col_inds = col_inds[row_inds == 32]
            interpolated_forwards = interpolated_forwards[row_inds == 32]

            #rays_o = rays_o[col_inds == 200]
            #rays_d = rays_d[col_inds == 200]
            #row_inds = row_inds[col_inds == 200]



            skip_rays = 8
            
            rays_d = rays_d[::skip_rays]
            rays_o = rays_o[::skip_rays]
            col_inds = col_inds[::skip_rays]
            row_inds = row_inds[::skip_rays]

            interpolated_forwards = interpolated_forwards[::skip_rays]

            #to cpu
            interpolated_forwards = interpolated_forwards.cpu().numpy()
            #remove z component
            interpolated_forwards = interpolated_forwards[:, :2]
            interpolated_forwards = interpolated_forwards / np.linalg.norm(interpolated_forwards, axis=1)[:, None]

 
    
            if False:
                torch_pcd = torch.tensor(big_pcd[:, :3]).to(device)
                intensities = torch.tensor(big_pcd[:, 3]).to(device)
                H, W = (ray_res_y ), ray_res_x

                # Create a simple RGB image on device (red background)
                img = torch.zeros((H, W, 3), device=device, dtype=torch.uint8)
                img[..., 0] = 128
                img[..., 2] = 128


                # Flatten ray origins/directions
                rays_o_flat = rays_o.view(-1, 3)
                rays_d_flat = rays_d.view(-1, 3)
                rays_d_norm = rays_d_flat / torch.norm(rays_d_flat, dim=-1, keepdim=True).clamp(min=1e-9)
                
                # Angle threshold in radians (e.g. ~3 degrees)
                angle_thresh = .1 * (3.14159 / 180.0)
                # Process points in chunks
                chunk_size = 1024 * 1024/ray_res_x * 64/ray_res_y
                #chunk to int
                chunk_size = int(chunk_size)
                for start in range(0, torch_pcd.shape[0], chunk_size):
                    print(start)
                    end = min(start + chunk_size, torch_pcd.shape[0])
                    chunk_points = torch_pcd[start:end]
            
                    # Vector from rays to points: [points, 1, 3] - [1, rays, 3] => [points, rays, 3]
                    vec = chunk_points[:, None, :] - rays_o_flat[None, :, :]
                    dist = torch.norm(vec, dim=-1).clamp(min=1e-9)
                    vec_norm = vec / dist.unsqueeze(-1)

                    # Dot product and angle check
                    dot = (vec_norm * rays_d_norm[None, ...]).sum(dim=-1)
                    angle = torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
                    mask = angle < angle_thresh

                    # For each ray, pick the first point that passes angle check
                    # (mask.cumsum(dim=0) == 1) keeps only the first True in each column
                    first_valid_mask = mask & (mask.cumsum(dim=0) == 1)
                    point_idx, ray_idx = first_valid_mask.nonzero(as_tuple=True)

                    # Get intensities for those point indices and convert to 3-channel grayscale
                    pix_intensity = (intensities[start:end][point_idx] * 255).clamp(0, 255).byte()
                    img_h = ray_idx // W
                    img_w = ray_idx % W
                    pix_vals = torch.stack([pix_intensity, pix_intensity, pix_intensity], dim=1)

                    # Write into the image at once using advanced indexing
                    img[img_h, img_w] = pix_vals


                #write img
                img = img.cpu().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_proj/"+file.split(".")[0] + "_ray.png", img)

            rays_o = rays_o.cpu().numpy()
            rays_d = rays_d.cpu().numpy()
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #for origin, direction in zip(rays_o, rays_d):
            #    direction = direction *0.01
            #    ax.plot([origin[0], origin[0] + direction[0]],
            #            [origin[1], origin[1] + direction[1]],
            #            [origin[2], origin[2] + direction[2]], color='b', linewidth=0.5)
                
            if True:
            
                rays_d = rays_d[:, :2]
                #normalize
                rays_d = rays_d / np.linalg.norm(rays_d, axis=1)[:, None]*10
                # Plot 2D rays from rays_o to direction



                # Use existing figure, just add new plots
                q = plt.quiver(rays_o[:, 0], rays_o[:, 1],
                        rays_d[:, 0], rays_d[:, 1],
                        col_inds.cpu(),  # Color by column index
                        scale=20, width=0.001,
                        cmap='winter')  # Add colormap
                #draw dots for translation
                plt.plot([prev_trans[0], trans[0], next_trans[0]], 
                        [prev_trans[1], trans[1], next_trans[1]], 'k-', alpha=0.5)
                
                #plt.plot(trans[0], trans[1], 'ro')
                #plt.plot(prev_trans[0], prev_trans[1], 'go')  
                #plt.plot(next_trans[0], next_trans[1], 'bo')

                # Draw forward vectors as larger arrows
                plt.quiver([prev_trans[0], trans[0], next_trans[0]], 
                        [prev_trans[1], trans[1], next_trans[1]],
                        [prev_forward[0], forward[0], next_forward[0]], 
                        [prev_forward[1], forward[1], next_forward[1]],
                        color=['g','r','b'], scale=15, width=0.005)
                plt.quiver(rays_o[:, 0], rays_o[:, 1], interpolated_forwards[:, 0], interpolated_forwards[:, 1], color='y', scale=10, width=0.001)

            else:

                print(rays_d.shape)
            
                #leave out y, take x and z
                rays_d = rays_d[:, [0, 2]]
                #plot only y and z
                #normalize
                rays_d = rays_d / np.linalg.norm(rays_d, axis=1)[:, None]*20

                # Use existing figure, just add new plots
                for i in range(len(rays_o)):
                    plt.plot([rays_o[i, 1], rays_o[i, 1] + rays_d[i, 0]], 
                             [rays_o[i, 2], rays_o[i, 2] + rays_d[i, 1]], 
                             color='blue', linewidth=0.5, alpha=0.5)
                
                plt.plot(rays_o[:, 1], rays_o[:, 2], 'ro')

                #lim from 80 to 120 for y and -1 to 20 for x
       

            #
       
            # Don't call plt.show() here - it will be called after the loop
            #take 10 percent randomly of bin_pcd
            #indices = np.random.choice(bin_pcd.shape[0], int(bin_pcd.shape[0]*0.1), replace=False)
            #add points from bin_pcd
            #bin_pcd = bin_pcd[indices]
            #ax.scatter(bin_pcd[:,0], bin_pcd[:,1], bin_pcd[:,2], c='r', marker='o', s=1)
            
            #increae size of fig
            #ax.set_xlim([-0, 80])
            #ax.set_ylim([-40, 40])
            #ax.set_zlim([-20, 10])
        break

plt.colorbar(q)  # Add colorbar
plt.axis('equal')
plt.grid(True)


#plt.show()
#save
plt.savefig("debug_proj/rays.png")
print("saved to debug_proj/rays.png")

