# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import os
import torch
import numpy as np
import configargparse

from model.lidar4d import LiDAR4D
from model.runner import Trainer
from utils.metrics import DepthMeter, IntensityMeter, RaydropMeter, PointsMeter
from utils.misc import set_seed


def get_arg_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, default="configs/kitti360_4950.txt", help="config file path")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--refine", action="store_true", help="refine mode")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--test_eval", action="store_true", help="test and eval mode")
    parser.add_argument("--seed", type=int, default=0)

    ### dataset
    parser.add_argument("--dataloader", type=str, choices=("kitti360", "nuscenes"), default="kitti360")
    parser.add_argument("--path", type=str, default="data/kitti360", help="dataset root path")
    parser.add_argument("--sequence_id", type=str, default="4950")
    parser.add_argument("--preload", type=bool, default=True, help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument("--bound", type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3")
    parser.add_argument("--scale", type=float, default=0.01, help="scale lidar location into box[-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of lidar location")
    parser.add_argument("--z_offsets" , type=float, nargs="*", default=[0, 0], help="offset of bottom lidar location")
    parser.add_argument("--laser_offsets" , type=float, nargs="*", default=0, help="offset of lasers")
    parser.add_argument("--near_lidar", type=float, default=1.0, help="minimum near distance for lidar")
    parser.add_argument("--far_lidar", type=float, default=81.0, help="maximum far distance for lidar")
    parser.add_argument("--fov_lidar", type=float, nargs="*", default=[2.0, 26.9], help="fov up and fov range of lidar")
    parser.add_argument("--num_frames", type=int, default=51, help="total number of sequence frames")
    parser.add_argument("--experiment_name", type=str, default="lidar4d", help="experiment name")
    

    ### LiDAR4D
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

    #list of opt params
    parser.add_argument("--opt_params", type=str, nargs="*", default=["laser_strengths"])#, "z_offsets", "fov_lidar", "laser_offsets", "R", "T"], help="list of opt params")
    parser.add_argument("--lr_factors", type=float, nargs="*", default=[0.1])#, 0.001, 0.001, 0.01, 0.01, 0.01], help="list of lr factors")

    ### training
    parser.add_argument("--depth_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--depth_grad_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--intensity_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--raydrop_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--flow_loss", type=bool, default=False)
    parser.add_argument("--grad_loss", type=bool, default=True)

    parser.add_argument("--alpha_d", type=float, default=1)
    parser.add_argument("--alpha_i", type=float, default=0.1)
    parser.add_argument("--alpha_r", type=float, default=0.01)
    parser.add_argument("--alpha_grad", type=float, default=0.1)
    parser.add_argument("--alpha_grad_norm", type=float, default=0.1)
    parser.add_argument("--alpha_spatial", type=float, default=0.1)
    parser.add_argument("--alpha_tv", type=float, default=0.1)

    parser.add_argument("--grad_norm_smooth", action="store_true")
    parser.add_argument("--spatial_smooth", action="store_true")
    parser.add_argument("--tv_loss", action="store_true")
    parser.add_argument("--sobel_grad", action="store_true")
    parser.add_argument("--urf_loss", action="store_true", help="enable line-of-sight loss in URF.")
    parser.add_argument("--active_sensor", action="store_true", help="enable volume rendering for active sensor.")

    parser.add_argument("--density_scale", type=float, default=1)
    parser.add_argument("--intensity_scale", type=float, default=1)
    parser.add_argument("--raydrop_ratio", type=float, default=0.5)
    parser.add_argument("--smooth_factor", type=float, default=0.2)

    parser.add_argument("--iters", type=int, default=30000, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--fp16", type=bool, default=True, help="use amp mixed precision training")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--num_rays_lidar", type=int, default=1024, help="num rays sampled per image for each training step")
    parser.add_argument("--num_steps", type=int, default=768, help="num steps sampled per ray")
    parser.add_argument("--patch_size_lidar", type=int, default=1, help="[experimental] render patches in training." 
                                                                        "1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument("--change_patch_size_lidar", nargs="+", type=int, default=[2, 8], help="[experimental] render patches in training. " 
                                                                      "1 means disabled, use [64, 32, 16] to enable, change during training")
    parser.add_argument("--change_patch_size_epoch", type=int, default=2, help="change patch_size intenvel")
    parser.add_argument("--ema_decay", type=float, default=0.95, help="use ema during training")

    return parser

def calculate_velocity(positions, times):
    """
    Calculate velocities for a sequence of positions using central difference.

    Parameters:
    positions (list of float): Sequence of positions.
    delta_t (float): Time interval between consecutive positions.

    Returns:
    list of float: Sequence of velocities.
    """
    n = len(positions)
    if n < 2:
        raise ValueError("At least two positions are required to calculate velocity.")

    velocities = []

    #multiply times by n and divide by 0.1
    times = times * n * 0.1

    # Calculate velocity for the first point (forward difference)
    v0 = (positions[1] - positions[0]) / (times[1] - times[0]) 

    velocities.append(v0)

    # Calculate velocity for internal points (central difference)
    for i in range(1, n - 1):
        v = (positions[i + 1] - positions[i - 1]) / (times[i + 1] - times[i - 1])
        velocities.append(v)

    # Calculate velocity for the last point (backward difference)
    vn = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
    velocities.append(vn)

    #list to tensor
    velocities = torch.stack(velocities)

    return velocities

def num_frames_from_sequence_id(sequence_id):
    if sequence_id == "1538":
        print("Using sequence 1538-1601")
        frame_start = 1538
        frame_end = 1601
    elif sequence_id == "1728":
        print("Using sequence 1728-1791")
        frame_start = 1728
        frame_end = 1791
    elif sequence_id == "1908":
        print("Using sequence 1908-1971")
        frame_start = 1908
        frame_end = 1971
    elif sequence_id == "3353":
        print("Using sequence 3353-3416")
        frame_start = 3353
        frame_end = 3416
    
    elif sequence_id == "2350":
        print("Using sequence 2350-2400")
        frame_start = 2350
        frame_end = 2400
    elif sequence_id == "4950":
        print("Using sequence 4950-5000")
        frame_start = 4950
        frame_end = 5000
    elif sequence_id == "8120":
        print("Using sequence 8120-8170")
        frame_start = 8120
        frame_end = 8170
    elif sequence_id == "10200":
        print("Using sequence 10200-10250")
        frame_start = 10200
        frame_end = 10250
    elif sequence_id == "10750":
        print("Using sequence 10750-10800")
        frame_start = 10750
        frame_end = 10800
    elif sequence_id == "11400":
        print("Using sequence 11400-11450")
        frame_start = 11400
        frame_end = 11450
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")
    
    return frame_end - frame_start + 1

def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    set_seed(opt.seed)
    torch.cuda.empty_cache() 
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    #assert opt_params and lr_factors have the same length
    assert len(opt.opt_params) == len(opt.lr_factors), "opt_params and lr_factors should have the same length"

    # Check sequence id.
    kitti360_sequence_ids = [
        "1538",
        "1728",
        "1908",
        "3353",
        "2350",
        "4950",
        "8120",
        "10200",
        "10750",
        "11400",
    ]

    # Specify dataloader class
    if opt.dataloader == "kitti360":
        from data.kitti360_dataset import KITTI360Dataset as NeRFDataset

        if opt.sequence_id not in kitti360_sequence_ids:
            raise ValueError(
                f"Unknown sequence id {opt.sequence_id} for {opt.dataloader}"
            )
    # elif opt.dataloader == "nuscenes":
    #     from data.nus_dataset import NusDataset as NeRFDataset
    else:
        raise RuntimeError("Should not reach here.")

    # Logging
    os.makedirs(opt.workspace, exist_ok=True)
    f = os.path.join(opt.workspace, "args.txt")
    with open(f, "w") as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write("{} = {}\n".format(arg, attr))

    if opt.patch_size_lidar > 1:
        assert (
            opt.num_rays % (opt.patch_size_lidar**2) == 0
        ), "patch_size ** 2 should be dividable by num_rays."

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
    # print(model)
    print(opt)
    
    loss_dict = {
        "mse": torch.nn.MSELoss(reduction="none"),
        "l1": torch.nn.L1Loss(reduction="none"),
        "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
        "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
        "cos": torch.nn.CosineSimilarity(),
    }
    criterion = {
        "depth": loss_dict[opt.depth_loss],
        "raydrop": loss_dict[opt.raydrop_loss],
        "intensity": loss_dict[opt.intensity_loss],
        "grad": loss_dict[opt.depth_grad_loss],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    laser_lines = 64
    if opt.laser_offsets == 0 or len(opt.laser_offsets) != laser_lines:
        print("laser_offsets is 0 or not the same length as laser_lines")
        laser_offsets = torch.zeros(laser_lines)
    else:
        laser_offsets = np.array(opt.laser_offsets).astype(np.float32)




    
    #opt.z_offsets = [0.2,0.12]
    #opt.fov_lidar = [2.0, 11, -11.45, 16]
    opt.z_offsets = torch.nn.Parameter(torch.tensor(opt.z_offsets).to(device))
    opt.fov_lidar = torch.nn.Parameter(torch.tensor(opt.fov_lidar).to(device))
    opt.laser_offsets = torch.nn.Parameter(torch.tensor(laser_offsets).to(device))


    
    lidar_metrics = [
        RaydropMeter(ratio=opt.raydrop_ratio),
        IntensityMeter(scale=opt.intensity_scale),
        DepthMeter(scale=opt.scale),
        PointsMeter(scale=opt.scale, intrinsics=opt.fov_lidar, z_offsets=opt.z_offsets, laser_offsets=opt.laser_offsets),
    ]


    
    num_frames = num_frames_from_sequence_id(opt.sequence_id)
    #pose_offsets R and T for each frame
    opt.R = torch.zeros((num_frames, 3))
    #R = torch.rand((num_frames, 3), requires_grad=True)*0.4-0.2
    opt.T = torch.zeros((num_frames, 3))
    #random between -0.1 and 0.1
    #T = torch.rand((num_frames, 3), requires_grad=True)*0.01-0.005
    #nn parameters
    opt.R = torch.nn.Parameter(opt.R.to(device))
    opt.T = torch.nn.Parameter(opt.T.to(device))

    opt.laser_strengths = torch.zeros((laser_lines, 2))
    #multiply first channel by 2
    opt.laser_strengths[:,  0] = 1#laser_strengths[:, 0] *2
    opt.laser_strengths = torch.nn.Parameter(opt.laser_strengths.to(device))



    if opt.test or opt.test_eval or opt.refine:
        trainer = Trainer(
            opt.experiment_name,
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            lidar_metrics=lidar_metrics,
            use_checkpoint=opt.ckpt,
            laser_strength  = opt.laser_strengths,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        )


        opt.z_offsets = trainer.z_offsets
        opt.fov_lidar = trainer.fov_lidar
        opt.laser_offsets = trainer.laser_offsets
        opt.R = trainer.R
        opt.T = trainer.T


        if opt.refine: # optimize raydrop only
            refine_loader = NeRFDataset(
                device=device,
                split="refine",
                root_path=opt.path,
                sequence_id=opt.sequence_id,
                preload=opt.preload,
                scale=opt.scale,
                offset=opt.offset,
                fp16=opt.fp16,
                patch_size_lidar=opt.patch_size_lidar,
                num_rays_lidar=opt.num_rays_lidar,
                fov_lidar=opt.fov_lidar,
                z_offsets=opt.z_offsets,
                laser_offsets=opt.laser_offsets,
                R = opt.R,
                T = opt.T,
            ).dataloader()

            trainer.refine(refine_loader)

        test_loader = NeRFDataset(
            device=device,
            split="val",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        ).dataloader()

        if test_loader.has_gt and not opt.test:
            trainer.evaluate(test_loader)

        trainer.test(test_loader, write_video=False)

    else:  # full pipeline


        if False:
            num_frames = train_loader._data.__len__()
            poses = train_loader._data.poses_lidar


            ref_pose = poses[0]
            inv_ref_pose = torch.linalg.inv(ref_pose)
            poses = inv_ref_pose @ poses
            lidar_times = train_loader._data.times
            positions = poses[:, :3, 3]
            velocity = calculate_velocity(positions,lidar_times)
            #velocity = torch.zeros((num_frames, 3))
            #magnitude of each 3d velocity
            cpu_velocity = velocity.cpu().detach().numpy()/opt.scale*3.6
            cpu_velocity = np.linalg.norm(cpu_velocity, axis=1)

            print("mean velocity", cpu_velocity.mean())
            velocity = torch.nn.Parameter(velocity.to(device))



        params = model.get_params(opt.lr)
        for param, lr_factor in zip(opt.opt_params, opt.lr_factors):
            if lr_factor > 0:
                params.append({"params": [getattr(opt, param)], "lr": lr_factor * opt.lr})

  
        optimizer = lambda model: torch.optim.Adam(
            params,
            #+ [{"params": [laser_strengths], "lr": 0.1 * opt.lr}] 
            #+ [{"params": [opt.z_offsets], "lr": 0.001 * opt.lr}] 
            #+ [{"params" :[opt.fov_lidar], "lr": 0.001* opt.lr}]
            #
            #+ [{"params": [R], "lr": 0.01 * opt.lr}]
            #+ [{"params": [T], "lr": 0.01 * opt.lr}]
            #+ [{"params": [laser_offsets], "lr": 0.01 * opt.lr}]
             
            betas=(0.9, 0.99),
            eps=1e-15
        )

  

        
        




        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
        )

        

        trainer = Trainer(
            opt.experiment_name,
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            lidar_metrics=lidar_metrics,
            use_checkpoint=opt.ckpt,
            optimizer=optimizer,
            ema_decay=opt.ema_decay,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            eval_interval=opt.eval_interval,
            laser_strength  = opt.laser_strengths,
            fov_lidar = opt.fov_lidar,
            z_offsets = opt.z_offsets,
            laser_offsets = opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        )


        opt.z_offsets = trainer.z_offsets
        opt.fov_lidar = trainer.fov_lidar
        opt.laser_offsets = trainer.laser_offsets
        opt.R = trainer.R
        opt.T = trainer.T


        train_loader = NeRFDataset(
            device=device,
            split="train",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        ).dataloader()

        valid_loader = NeRFDataset(
            device=device,
            split="val",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        ).dataloader()

        # optimize raydrop
        refine_loader = NeRFDataset(
            device=device,
            split="refine",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        ).dataloader()


        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print(f"max_epoch: {max_epoch}")
        trainer.train(train_loader, valid_loader, refine_loader, max_epoch)


        # also test
        '''
        test_loader = NeRFDataset(
            device=device,
            split="test",
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size_lidar=opt.patch_size_lidar,
            num_rays_lidar=opt.num_rays_lidar,
            fov_lidar=opt.fov_lidar,
            z_offsets=opt.z_offsets,
            laser_offsets=opt.laser_offsets,
            R = opt.R,
            T = opt.T,
        ).dataloader()
        '''
        test_loader = valid_loader
        #if test_loader.has_gt:
        #    trainer.evaluate(test_loader)  # evaluate metrics

        #trainer.test(test_loader, write_video=False)  # save final results



if __name__ == "__main__":
    main()
