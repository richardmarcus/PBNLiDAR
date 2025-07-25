import os
import math
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import configargparse
from pathlib import Path
from packaging import version as pver
from data.preprocess.kitti360_loader import KITTI360Loader
from model.lidar4d import LiDAR4D
from model.simulator import Simulator
from utils.misc import set_seed
from main_pbl import num_frames_from_sequence_id


def get_arg_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, default="configs/kitti360_4950.txt", help="config file path")
    parser.add_argument("--workspace", type=str, default="simulation")
    parser.add_argument("--ckpt", type=str, default="latest_model", help="path of trained model weight")
    parser.add_argument("--seed", type=int, default=0)

    ### dataset (keep the same as training)
    parser.add_argument("--dataloader", type=str, choices=("kitti360", "nuscenes"), default="kitti360")
    parser.add_argument("--path", type=str, default="data/kitti360", help="dataset root path")
    parser.add_argument("--sequence_id", type=str, default="4950")
    parser.add_argument("--preload", type=bool, default=True, help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument("--bound", type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3")
    parser.add_argument("--scale", type=float, default=0.01, help="scale lidar location into box[-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of lidar location")
    parser.add_argument("--near_lidar", type=float, default=1.0, help="minimum near distance for lidar")
    parser.add_argument("--far_lidar", type=float, default=81.0, help="maximum far distance for lidar")
    parser.add_argument("--num_frames", type=int, default=51, help="total number of sequence frames")
    parser.add_argument("--active_sensor", action="store_true", help="enable volume rendering for active sensor.")
    parser.add_argument("--density_scale", type=float, default=1)
    parser.add_argument("--fp16", type=bool, default=True, help="use amp mixed precision training")
    parser.add_argument("--num_steps", type=int, default=768, help="num steps sampled per ray")

    ### LiDAR4D (keep the same as training)
    parser.add_argument("--min_resolution", type=int, default=32, help="minimum resolution for planes")
    parser.add_argument("--base_resolution", type=int, default=512, help="minimum resolution for hash grid")
    parser.add_argument("--max_resolution", type=int, default=32768, help="maximum resolution for hash grid")
    parser.add_argument("--time_resolution", type=int, default=8, help="temporal resolution")
    parser.add_argument("--n_levels_plane", type=int, default=4, help="n_levels for planes")
    parser.add_argument("--n_features_per_level_plane", type=int, default=8, help="n_features_per_level for planes")
    parser.add_argument("--n_levels_hash", type=int, default=8, help="n_levels for hash grid")
    parser.add_argument("--n_features_per_level_hash", type=int, default=4, help="n_features_per_level for hash grid")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="hashmap size for hash grid")
    parser.add_argument("--num_layers_flow", type=int, default=3, help="num_layers of flownet")
    parser.add_argument("--hidden_dim_flow", type=int, default=64, help="hidden_dim of flownet")
    parser.add_argument("--num_layers_sigma", type=int, default=2, help="num_layers of sigmanet")
    parser.add_argument("--hidden_dim_sigma", type=int, default=64, help="hidden_dim of sigmanet")
    parser.add_argument("--geo_feat_dim", type=int, default=15, help="geo_feat_dim of sigmanet")
    parser.add_argument("--num_layers_lidar", type=int, default=3, help="num_layers of intensity/raydrop")
    parser.add_argument("--hidden_dim_lidar", type=int, default=64, help="hidden_dim of intensity/raydrop")
    parser.add_argument("--out_lidar_dim", type=int, default=2, help="output dim for lidar intensity/raydrop")
    parser.add_argument("--use_refine", type=bool, default=True, help="use ray-drop refinement")

    ### simulation
    parser.add_argument("--use_camera", action="store_true", help="use camera for simulation")
    parser.add_argument("--use_cam_poses", action="store_true", help="use camera poses for lidar rays")
    parser.add_argument("--fov_lidar", type=float, nargs="*", default=[2.0, 13.45, -11.45, 13.45], help="fov up and fov range of lidar")
    parser.add_argument("--H_lidar", type=int, default=66, help="height of lidar range map")
    parser.add_argument("--W_lidar", type=int, default=1030, help="width of lidar range map")
    parser.add_argument("--experiment_name", type=str, default="lidar4d", help="experiment name")
    parser.add_argument("--laser_offsets" , type=float, nargs="*", default=0, help="offset of lasers")
    

    parser.add_argument("--shift_x", type=float, default=0.0, help="translation on x direction (m)")
    parser.add_argument("--shift_y", type=float, default=0.0, help="translation on y direction (m)")
    parser.add_argument("--shift_z", type=float, default=0.0, help="translation on z direction (m)")
    parser.add_argument("--shift_z_bottom", type=float, default=0.0, help="translation on z direction (m) for bottom lidar")
    parser.add_argument("--shift_z_top", type=float, default=0.0, help="translation on z direction (m) for top lidar")
    parser.add_argument("--align_axis", action="store_true", help="align shift axis to vehicle motion direction.")
    parser.add_argument("--kitti2nus", action="store_true", help="a simple demo to change lidar configuration from kitti360 to nuscenes.")
    parser.add_argument("--interpolation_factor", type=float, default=0.0, help="interpolation factor for lidar2world")
    parser.add_argument("--motion", type=bool, default=False, help="use motion correction (rolling shutter)")
    #list of opt params
    parser.add_argument("--opt_params", type=str, nargs="*", default=[]#"R", "T"]#"laser_strength"]#, "near_range_threshold", "near_range_factor", "distance_scale", "near_offset","distance_fall"])#, "z_offsets", "fov_lidar", "laser_offsets", "R", "T"], help="list of opt params"
                        )
    parser.add_argument("--lr_factors", type=float, nargs="*", default=[]#0.01,0.01]#0.1]#,0.05,0.05,0.002,0.1,0.1])#, 0.001, 0.001, 0.01, 0.01, 0.01], help="list of lr factors"
                        )

    return parser


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def _get_frame_ids(sequence_id):
    # For KITTI-360
    if sequence_id == "1538":
        s_frame_id = 1538
        e_frame_id = 1601
    elif sequence_id == "1728":
        s_frame_id = 1728
        e_frame_id = 1791
    elif sequence_id == "1908":
        s_frame_id = 1908
        e_frame_id = 1971
    elif sequence_id == "3353":
        s_frame_id = 3353
        e_frame_id = 3416

    elif sequence_id == "2350":
        s_frame_id = 2350
        e_frame_id = 2400
    elif sequence_id == "4950":
        s_frame_id = 4950
        e_frame_id = 5000
    elif sequence_id == "8120":
        s_frame_id = 8120
        e_frame_id = 8170
    elif sequence_id == "10200":
        s_frame_id = 10200
        e_frame_id = 10250
    elif sequence_id == "10750":
        s_frame_id = 10750
        e_frame_id = 10800
    elif sequence_id == "11400":
        s_frame_id = 11400
        e_frame_id = 11450
    else:
        s_frame_id = int(sequence_id)
        e_frame_id = s_frame_id + 50
        #raise ValueError(f"Invalid sequence id: {sequence_id}")

    return s_frame_id, e_frame_id


def _get_camera_rays(sequence_id, opt, device, step=4):
    # For KITTI-360
    kitti_360_root = Path(opt.path) / "KITTI-360"
    sequence_name = "2013_05_28_drive_"+opt.scene_id
    s_frame_id, e_frame_id = _get_frame_ids(sequence_id)
    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    print(f"Simulation using sequence {s_frame_id}-{e_frame_id}")

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get camera2world.
    camera2world_direct, intrinsics, extriniscs, rectification = k3.load_cameras(sequence_name, frame_ids)


    # Offset and scale
    poses = np.stack(camera2world_direct, axis=0)
    poses[:, :3, -1] = (poses[:, :3, -1] - opt.offset) * opt.scale
    poses = torch.from_numpy(poses).to(device).float()  # [N, 4, 4]

    # Get directions based on H, W and intrinsics
    B = poses.shape[0]
    H = opt.H_lidar
    W = opt.W_lidar
    # Allow taking steps in the linspace
    step_w = step  
    step_h = step 

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W//step_w, device=device), 
        torch.linspace(0, H - 1, H//step_h, device=device),
    )  # float
    
    # Adjust reshape dimensions for the new grid size
    new_H = H//step_h
    new_W = W//step_w
    i = i.t().reshape([1, new_H * new_W]).expand([B, new_H * new_W])
    j = j.t().reshape([1, new_H * new_W]).expand([B, new_H * new_W])

  
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]  
    cy = intrinsics[1, 2]

 
    # Convert pixel coordinates to normalized device coordinates
    x = (i+0.5 - cx) / fx
    y = (j+0.5 - cy) / fy


    # For pinhole camera model, directions point from origin through image plane points
    directions = torch.stack(
        [
            x,  # x-coordinate on image plane
            y,  # y-coordinate on image plane
            torch.ones_like(x)  # z=1 plane
        ],
        -1,
    )
    # Normalize the direction vectors
    directions = F.normalize(directions, dim=-1)


    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]


    times_lidar = []
    for frame in frame_ids:
        time = np.asarray((frame-s_frame_id)/(e_frame_id-s_frame_id))
        times_lidar.append(time)
    times_lidar = torch.from_numpy(np.asarray(times_lidar, dtype=np.float32)).view(-1, 1).to(device).float() # [N, 1]

    return rays_o, rays_d, times_lidar


def _get_lidar_rays(sequence_id, opt, device, interpolation, cam_poses=False,shift_up=0, shift_down=0):
    # For KITTI-360
    kitti_360_root = Path(opt.path) / "KITTI-360"
    sequence_name = "2013_05_28_drive_"+opt.scene_id
    s_frame_id, e_frame_id = _get_frame_ids(sequence_id)
    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    print(frame_ids)
    print(f"Simulation using sequence {s_frame_id}-{e_frame_id}")

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids, interpolation, cam_poses=cam_poses)

    # Offset and scale
    poses = np.stack(lidar2world, axis=0)
    poses[:, :3, -1] = (poses[:, :3, -1] - opt.offset) * opt.scale
    poses = torch.from_numpy(poses).to(device).float()  # [N, 4, 4]

    # Get directions based on H, W and fov_lidar
    B = poses.shape[0]
    H = opt.H_lidar//2
    W = opt.W_lidar

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])

    fov_up, fov, fov_up2, fov2= opt.fov_lidar
    #fov*=2
    beta = -(i - W / 2) / W * 2 * np.pi

    alpha = (fov_up - j / H * fov) / 180 * np.pi

    directions = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )
    #print min max of alpha
    alpha = (fov_up2 - j / H * fov2) / 180 * np.pi
    directions_bottom = torch.stack(
        [
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta),
            torch.sin(alpha),
        ],
        -1,
    )


    #concatenate top and bottom lidar
    directions = torch.cat([directions, directions_bottom], dim=1)

    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]


    times_lidar = []
    for frame in frame_ids:
        time = np.asarray((frame-s_frame_id)/(e_frame_id-s_frame_id))
        times_lidar.append(time)
    times_lidar = torch.from_numpy(np.asarray(times_lidar, dtype=np.float32)).view(-1, 1).to(device).float() # [N, 1]

    return rays_o, rays_d, times_lidar


def main():
    parser = get_arg_parser()
    opt = parser.parse_args()


    #last four digits before the .txt of opt.config
    opt.scene_id = opt.config.split("/")[-1].split(".")[0][-4:]

    
   
    print(f"Config file: {opt.config}")
    print(f"Workspace directory: {opt.workspace}")
    print(f"Experiment name: {opt.experiment_name}")
    print(f"Dataset path: {opt.path}")
    print(f"Using motion correction: {opt.motion}")
    print(f"Checkpoint: {opt.ckpt}")
    print(f"Lidar output dimension: {opt.out_lidar_dim}")
    print(f"List of optimizer params: {opt.opt_params}")
    print(f"List of learning rate factors: {opt.lr_factors}")
    print(f"Sequence id: {opt.sequence_id}")
    print(f"Scene id: {opt.scene_id}")
    print("----------------------------------------")
  
    set_seed(opt.seed)

    # Logging
    os.makedirs(opt.workspace, exist_ok=True)

    # simple demo for lidar configuration from kitti360 to nuscenes
    if opt.kitti2nus:
        opt.fov_lidar = [10.0, 40.0]
        opt.H_lidar = 32
        opt.W_lidar = 1024
        opt.far_lidar = 70
        opt.shift_z += 0.1 * opt.scale
        opt.use_refine = False

    opt.near_lidar = opt.near_lidar * opt.scale
    opt.far_lidar = opt.far_lidar * opt.scale

    model = LiDAR4D(
        min_resolution=opt.min_resolution,
        base_resolution=opt.base_resolution,
        max_resolution=opt.max_resolution,
        time_resolution=opt.time_resolution,
        n_levels_plane=opt.n_levels_plane,
        n_features_per_level_plane=opt.n_features_per_level_plane,
        n_levels_hash=opt.n_levels_hash,
        n_features_per_level_hash=opt.n_features_per_level_hash,
        log2_hashmap_size=opt.log2_hashmap_size,
        num_layers_flow=opt.num_layers_flow,
        hidden_dim_flow=opt.hidden_dim_flow,
        num_layers_sigma=opt.num_layers_sigma,
        hidden_dim_sigma=opt.hidden_dim_sigma,
        geo_feat_dim=opt.geo_feat_dim,
        num_layers_lidar=opt.num_layers_lidar,
        hidden_dim_lidar=opt.hidden_dim_lidar,
        out_lidar_dim=opt.out_lidar_dim,
        num_frames=opt.num_frames,
        bound=opt.bound,
        near_lidar=opt.near_lidar,
        far_lidar=opt.far_lidar,
        density_scale=opt.density_scale,
        active_sensor=opt.active_sensor,
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_frames = num_frames_from_sequence_id(opt.sequence_id)

    R = torch.zeros((num_frames, 3))
    T = torch.zeros((num_frames, 3))

    laser_lines = 64
    laser_offsets = torch.zeros(laser_lines)
    laser_strength = torch.ones((laser_lines))


    sim = Simulator(
        "lidar4d",
        opt,
        model,
        device=device,
        workspace=opt.workspace,
        fp16=opt.fp16,
        use_checkpoint=opt.ckpt,
        H_lidar=opt.H_lidar,
        W_lidar=opt.W_lidar,
        use_refine=opt.use_refine,
        fov_lidar=opt.fov_lidar,
        laser_offsets=laser_offsets,
        R = R,
        T = T,
    )

    if not opt.use_camera:
        #extend laser_offsets  and laser_strength from laser_lines to h_lidar
        if opt.H_lidar > laser_lines:
            num_new = opt.H_lidar - laser_lines
            # get num_new, 2 random values between 0 and laser_lines
            random_indices = torch.randint(0, laser_lines, (num_new, 2))
            # random factor between 0 and 1
            random_factors = torch.rand((num_new))
            extra_laser_offsets = laser_offsets[random_indices[:, 0]] * random_factors + laser_offsets[random_indices[:, 1]] * (1 - random_factors)
            extra_laser_strength = laser_strength[random_indices[:, 0]] * random_factors + laser_strength[random_indices[:, 1]] * (1 - random_factors)
            
            # Total length after extension
            total_length = laser_lines + num_new
            # Create interleaved indices
            original_indices = torch.linspace(0, total_length - 1, steps=laser_lines).long()

            #extra indices are all indices that are not in original_indices
            extra_indices = torch.tensor([i for i in range(total_length) if i not in original_indices])

            # Create new tensors for interleaving
            new_laser_offsets = torch.zeros(total_length, dtype=laser_offsets.dtype, device=laser_offsets.device)
            new_laser_strength = torch.zeros(total_length, dtype=laser_strength.dtype, device=laser_strength.device)

            # Assign original values
            new_laser_offsets[original_indices] = laser_offsets
            new_laser_strength[original_indices] = laser_strength

            # Assign extra values
            new_laser_offsets[extra_indices] = extra_laser_offsets
            new_laser_strength[extra_indices] = extra_laser_strength

            # Update the original tensors
            laser_offsets = new_laser_offsets
            laser_strength = new_laser_strength
        
        elif opt.H_lidar < laser_lines:
            laser_offsets = laser_offsets[:opt.H_lidar]
            laser_strength = laser_strength[:opt.H_lidar]



        sequence_id = opt.sequence_id

        # simulate novel configuration (e.g., fov_lidar, H_lidar, W_lidar)
        if opt.interpolation_factor > 0:
            interpolation = opt.interpolation_factor
        else:
            interpolation = None
        rays_o, rays_d, times_lidar = _get_lidar_rays(sequence_id, opt, device=device, interpolation=   interpolation, cam_poses=opt.use_cam_poses, shift_up=opt.shift_z_top, shift_down=opt.shift_z_bottom)

    else:
        step = 1
        rays_o, rays_d, times_lidar = _get_camera_rays(opt.sequence_id, opt, device=device, step=step)
        sim.W_lidar = opt.W_lidar // step
        sim.H_lidar = opt.H_lidar // step
        
        print("Camera rays")

    print(rays_o.shape, rays_d.shape, times_lidar.shape)

    # # simulate novel trajectory (global)
    # rays_o_shift = rays_o.clone()
    # rays_o_shift[:,:,0] = rays_o_shift[:,:,0] + opt.shift_x * opt.scale
    # rays_o_shift[:,:,1] = rays_o_shift[:,:,1] + opt.shift_y * opt.scale
    # rays_o_shift[:,:,2] = rays_o_shift[:,:,2] + opt.shift_z * opt.scale

    # simulate novel trajectory
    rays_o_shift = rays_o.clone()
    shift_x = opt.shift_x
    shift_y = opt.shift_y
    shift_z = opt.shift_z
    scale = opt.scale
    forward = torch.tensor([[1,0,0]]).to(rays_o)
    print(opt.shift_z_top, opt.shift_z_bottom)
    print(opt.fov_lidar)

    

    for i in range(rays_o.shape[0]):
        # align x axis to vehicle motion direction
        if opt.align_axis:
            if i < rays_o.shape[0] - 1:
                forward = F.normalize((rays_o[i+1,0,:] - rays_o[i,0,:]).unsqueeze(0), p=2)
            left = torch.tensor([-forward[:,1], forward[:,0], forward[:,2]]).to(forward)

            shift_x = (opt.shift_x * forward + opt.shift_y * left)[:, 0]
            shift_y = (opt.shift_x * forward + opt.shift_y * left)[:, 1]

            # # or you can set a sinusoidal trajectory
            # shift_x = (opt.shift_x * forward + opt.shift_y * left * math.sin(i/20*2*math.pi))[:, 0]
            # shift_y = (opt.shift_x * forward + opt.shift_y * left * math.sin(i/20*2*math.pi))[:, 1]

        rays_o_shift[i,:,0] = rays_o_shift[i,:,0] + shift_x * scale
        rays_o_shift[i,:,1] = rays_o_shift[i,:,1] + shift_y * scale
        rays_o_shift[i,:,2] = rays_o_shift[i,:,2] + shift_z * scale

        if not opt.use_camera:
            top_indices = torch.arange(0, opt.H_lidar//2 * opt.W_lidar).to(device)
            bot_indices = torch.arange(opt.H_lidar//2 * opt.W_lidar, opt.H_lidar * opt.W_lidar).to(device)
            rays_o_shift[i,top_indices,2] = rays_o_shift[i,top_indices,2] - opt.shift_z_top* scale
            rays_o_shift[i,bot_indices,2] = rays_o_shift[i,bot_indices,2] - opt.shift_z_bottom * scale
            #print(opt.shift_z_top, opt.shift_z_bottom)

    # save results
    sim.render(rays_o_shift, rays_d, times_lidar, save_pc=False)


if __name__ == "__main__":
    main()
