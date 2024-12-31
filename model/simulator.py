# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import os
import time
import glob
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from rich.console import Console
from utils.convert import pano_to_lidar, pano_to_lidar_with_intensities


class Simulator(object):
    def __init__(
        self,
        name,   # name of this experiment
        opt,    # extra conf
        model,  # network
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,   # whether to mute all print
        fp16=False,   # amp optimize level
        workspace="simulation",  # workspace to save logs & results
        use_checkpoint="latest_model",  # which ckpt to use at init time
        use_refine=True, # whether to use U-Net for refinement
        H_lidar=66,   # height of lidar range map
        W_lidar=1030, # width of lidar range map
        laser_strength = None,
        laser_offsets = None,
        alpha_offsets = None,
        fov_lidar = None,
        z_offsets = None,
        velocity = None,
        R = None,
        T = None,


    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.fp16 = fp16
        self.workspace = workspace
        self.use_checkpoint = use_checkpoint
        self.use_refine = use_refine
        self.H_lidar, self.W_lidar = H_lidar, W_lidar
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:0" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()
        self.laser_strength = laser_strength
        self.laser_offsets = laser_offsets
        self.velocity = velocity
        self.fov_lidar = fov_lidar
        self.z_offsets = z_offsets
        self.alpha_offsets = alpha_offsets
        self.R = R
        self.T = T

        model.to(self.device)
        self.model = model

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            # os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Simulator: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        # self.log(f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if not self.mute:
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file
    
    @torch.no_grad()
    def render(self, rays_o_lidar, rays_d_lidar, times_lidar, save_pc=True, save_img=True, save_video=True):
        if save_video:
            all_preds = []
            all_preds_depth = []
            all_preds_intensity = []
        B = rays_o_lidar.shape[0]

        optimized_laserstrength = self.laser_strength
        #rays_o_lidar, rays_d_lidar, optimized_laserstrength = self.load_sensor_settings(data, rays_o_lidar, rays_d_lidar)



        for i in tqdm.tqdm(range(B)):
            ray_o_lidar = rays_o_lidar[i, ...].unsqueeze(0)
            ray_d_lidar = rays_d_lidar[i, ...].unsqueeze(0)
            time_lidar = times_lidar[i, ...].unsqueeze(0)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs_lidar = self.model.render(
                    ray_o_lidar,
                    ray_d_lidar,
                    time_lidar,
                    staged=True,
                    perturb=False,
                    **vars(self.opt),
                )

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, self.H_lidar, self.W_lidar, 2)
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, self.H_lidar, self.W_lidar)


            # if self.opt.raydrop_loss == 'bce':
            #     pred_raydrop = F.sigmoid(pred_raydrop)
            if self.use_refine:
                pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
                pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
            raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
    
            if self.laser_strength is not None:
                mult_laser = optimized_laserstrength[:,0].repeat(self.W_lidar,1).T.reshape(1, self.H_lidar, self.W_lidar)
                add_laser = optimized_laserstrength[:,1].repeat(self.W_lidar,1).T.reshape(1, self.H_lidar, self.W_lidar)

                pred_intensity = pred_intensity * mult_laser + add_laser
            #pred_intensity = pred_intensity * raydrop_mask
            #pred_depth = pred_depth * raydrop_mask

            pred_raydrop = pred_raydrop[0].detach().cpu().numpy()
            pred_depth = pred_depth[0].detach().cpu().numpy()
            pred_intensity = pred_intensity[0].detach().cpu().numpy()
            #z_offsets from opt.z_offset and opt.z_offset_bottom
            z_offsets = [self.opt.shift_z_top, self.opt.shift_z_bottom]
            pred_lidar = pano_to_lidar_with_intensities(
                pred_depth / self.opt.scale, pred_intensity, self.opt.fov_lidar, z_offsets
            )

            if save_pc:
                save_path = os.path.join(
                        self.workspace,
                        "points",
                        f"lidar4d_{i:04d}.npy",
                    )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, pred_lidar)

            if save_img:
                save_path = os.path.join(
                        self.workspace,
                        "images",
                        f"lidar4d_{i:04d}.png",
                    )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                img_raydrop = (pred_raydrop * 255).astype(np.uint8)
                img_raydrop = cv2.cvtColor(img_raydrop, cv2.COLOR_GRAY2BGR)

                img_intensity = (pred_intensity * 255).astype(np.uint8)
                img_intensity = cv2.applyColorMap(img_intensity, 1)

                #save depth as float npy
                save_path_depth = os.path.join(
                        self.workspace,
                        "images",
                        f"lidar4d_{i:04d}.npy",
                    )
                #save path exists already
                np.save(save_path_depth, pred_depth)
                
                img_depth = (pred_depth * 255).astype(np.uint8)
                img_depth = cv2.applyColorMap(img_depth, 20)

                img_pred = cv2.vconcat([img_raydrop, img_intensity, img_depth])
                cv2.imwrite(save_path, img_pred)
              
            if save_video:
                img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
                all_preds.append(img_pred)

        if save_video:
            save_path = os.path.join(
                        self.workspace,
                        "video",
                        f"lidar4d_sim.mp4",
                    )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_preds = np.stack(all_preds, axis=0)

            imageio.mimwrite(
                save_path,
                all_preds,
                fps=5, # change frame rate here
                quality=8,
                macro_block_size=1,
            )

        return pred_lidar


    def load_checkpoint(self, checkpoint=None, model_only=True):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        
        
        if "laser_strength" in checkpoint_dict:
            self.laser_strength = torch.nn.Parameter(checkpoint_dict['laser_strength'].to(self.device))
            print("laser_strength loaded")
        else:
            print("laser_strength not found in checkpoint")
        if "z_offsets" in checkpoint_dict:
            self.z_offsets = torch.nn.Parameter(checkpoint_dict['z_offsets'].to(self.device))
            print("z_offsets loaded")

        if "fov_lidar" in checkpoint_dict:
            self.fov_lidar = torch.nn.Parameter(checkpoint_dict['fov_lidar'].to(self.device))
            print("fov_lidar loaded")

        if "alpha_offsets" in checkpoint_dict:
            self.alpha_offsets = torch.nn.Parameter(checkpoint_dict['alpha_offsets'].to(self.device))
            print("alpha_offsets loaded")

        if "laser_offsets" in checkpoint_dict:
            self.laser_offsets = torch.nn.Parameter(checkpoint_dict['laser_offsets'].to(self.device))
            print("laser_offsets loaded")

        if "velocity" in checkpoint_dict:
            self.velocity = torch.nn.Parameter(checkpoint_dict['velocity'].to(self.device))
            print("velocity loaded")

        if "R" in checkpoint_dict:
            self.R = torch.nn.Parameter(checkpoint_dict['R'].to(self.device))
            print("R loaded")

        if "T" in checkpoint_dict:
            self.T = torch.nn.Parameter(checkpoint_dict['T'].to(self.device))
            print("T loaded")

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if model_only:
            return

        if "stats" in checkpoint_dict:
            self.stats = checkpoint_dict["stats"]
        if "epoch" in checkpoint_dict:
            self.epoch = checkpoint_dict["epoch"]
        if "global_step" in checkpoint_dict:
            self.global_step = checkpoint_dict["global_step"]
            self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
