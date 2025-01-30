# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import glob
import os
import random
import time

import cv2
import imageio
from matplotlib import pyplot as plt
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
from utils.misc import point_removal
from data.kitti360_dataset import vec2skew, Exp
import kornia

from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

def slerp0(q1, q2, t):
    # Compute the cosine of the angle between the quaternions
    cos_theta = torch.sum(q1 * q2, dim=-1, keepdim=True)
    
    # If cos_theta < 0, negate one quaternion to ensure shortest path
    q2 = torch.where(cos_theta < 0, -q2, q2)
    cos_theta = torch.abs(cos_theta)
    
    # If quaternions are very close, use linear interpolation
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    
    # Avoid division by zero
    ratio1 = torch.where(sin_theta > 1e-6,
                        torch.sin((1 - t) * theta) / sin_theta,
                        1 - t)
    ratio2 = torch.where(sin_theta > 1e-6,
                        torch.sin(t * theta) / sin_theta,
                        t)
    
    return ratio1 * q1 + ratio2 * q2
def slerp(v0: FloatTensor, v1: FloatTensor, t: FloatTensor, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        DOT_THRESHOLD: Threshold for considering the two vectors as
                                colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    '''
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0, dim=-1)
    v1_norm: FloatTensor = norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: FloatTensor = lerp(v0, v1, t)

        out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():

        # Calculate initial angle between v0 and v1
        theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
        sin_theta_0: FloatTensor = theta_0.sin()
        # Angle at timestep t
        theta_t: FloatTensor = theta_0 * t
        sin_theta_t: FloatTensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: FloatTensor = sin_theta_t / sin_theta_0
        slerped: FloatTensor = s0 * v0 + s1 * v1

        out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
  
    return out

descriptors = ["mult_strength", "add_strength", "x_offset", "y_offset", "z_offset", "x_dir", "y_dir", "z_dir"]

def interpolate_poses(pose, pose_before, pose_after, num_steps=1024):
    # Generate interpolation factors
    interpolation_factors = torch.linspace(.5, 1, steps=num_steps//2, device=pose.device).view(-1, 1)  # Shape: [num_steps, 1]

    # Decompose the 4x4 matrices into rotation (3x3) and translation (3x1)
    rotation_before = pose_before[:, :3, :3]  # [B, 3, 3]
    rotation_after = pose_after[:, :3, :3]  # [B, 3, 3]
    rotation = pose[:, :3, :3]  # [B, 3, 3]
    translation = pose[:, :3, 3]  # [B, 3]
    translation_before = pose_before[:, :3, 3]  # [B, 3]
    translation_after = pose_after[:, :3, 3]  # [B, 3]

    # Convert rotations to quaternions for SLERP interpolation
    quaternion_before = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation_before)  # [B, 4]
    quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation)  # [B, 4]
    quaternion_after = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation_after)  # [B, 4]

    # SLERP interpolation for rotations
    quaternions_interpolated_before = slerp(quaternion_before.unsqueeze(1), quaternion.unsqueeze(1),interpolation_factors)  # [B, num_steps, 4]

    quaternions_interpolated_after = slerp(quaternion.unsqueeze(1), quaternion_after.unsqueeze(1), interpolation_factors-0.5)  # [B, num_steps, 4]

    #concate both
    quaternions_interpolated = torch.cat([quaternions_interpolated_before, quaternions_interpolated_after], dim=1)

    # Convert interpolated quaternions back to rotation matrices
    rotations_interpolated = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternions_interpolated)  # [B, num_steps, 3, 3]

    # Linear interpolation for translations

    translations_interpolated_before = translation.unsqueeze(1) * (interpolation_factors) + \
                                translation_before.unsqueeze(1) * (1-interpolation_factors)  # [B, num_steps, 3]
    
    translations_interpolated_after = translation.unsqueeze(1) * (1.5-interpolation_factors) + translation_after.unsqueeze(1) * (interpolation_factors-.5)  # [B, num_steps, 3]
    
    translations_interpolated = torch.cat([translations_interpolated_before, translations_interpolated_after], dim=1)

    # Combine interpolated rotations and translations into 4x4 matrices
    poses_interpolated = torch.eye(4, device=pose.device).repeat(pose_before.size(0), num_steps, 1, 1)  # [B, num_steps, 4, 4]
    poses_interpolated[:, :, :3, :3] =rotations_interpolated  # Set rotations
    poses_interpolated[:, :, :3, 3] = translations_interpolated  # Set translations


    

    return poses_interpolated


# Function to compute the rotation matrix from a batch of axis-angle vectors
def axis_angle_vector_to_rotation_matrix(axis_angle_vectors):
    # Compute angles and normalize axes
    angles = np.linalg.norm(axis_angle_vectors, axis=1)  # Magnitude gives the angle for each vector
    axes = axis_angle_vectors / angles[:, np.newaxis]  # Normalize each axis

    # Extract the components of each axis (x, y, z)
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    t = 1 - c

    # Rodrigues' rotation formula applied to each axis-angle vector
    # Resulting rotation matrix will have shape (n, 3, 3)
    R = np.zeros((axes.shape[0], 3, 3))

    R[:, 0, 0] = t*x*x + c
    R[:, 0, 1] = t*x*y - s*z
    R[:, 0, 2] = t*x*z + s*y
    R[:, 1, 0] = t*x*y + s*z
    R[:, 1, 1] = t*y*y + c
    R[:, 1, 2] = t*y*z - s*x
    R[:, 2, 0] = t*x*z - s*y
    R[:, 2, 1] = t*y*z + s*x
    R[:, 2, 2] = t*z*z + c

    return R

def plot_velocity(data, scale):
    """
    Plot velocity in 2D with color mapping and a normalized time series plot.

    Parameters:
    data (np.ndarray): Velocity data as a 2D array.
    scale (float): Scale factor to adjust velocities.
    """
    # Rescale velocities
    velocities = data / scale
    time_step = 0.1  # 10 Hz => 0.1s per frame

    os.makedirs("velocity", exist_ok=True)

    # Initialize positions array
    positions = np.zeros((velocities.shape[0] + 1, 3))  # 1 extra for the initial position

    # Calculate positions by integrating velocity
    for i in range(velocities.shape[0]):
        positions[i + 1] = positions[i] + velocities[i] * time_step

    # 2D Plot: X and Y positions with velocity magnitude as color
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:-1, 0], positions[:-1, 1], c=velocity_magnitude, cmap='viridis', label='Trajectory')
    plt.colorbar(label='Velocity Magnitude')
    plt.title("Vehicle Trajectory (2D)", fontsize=14)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.savefig("velocity/trajectory_2d.png")

    max_x = positions[:, 0].max()
    min_x = positions[:, 0].min()
    max_y = positions[:, 1].max()
    min_y = positions[:, 1].min()
    max_z = positions[:, 2].max()
    min_z = positions[:, 2].min()

    # Normalize positions for the second plot
    normalized_positions = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0))

    # 2D Plot: X, Y, Z components normalized over time
    plt.figure(figsize=(10, 8))
    indices = np.arange(len(normalized_positions))
    plt.plot(indices, normalized_positions[:, 0], label=f'Normalized X (min: {min_x:.2f}, max: {max_x:.2f})', color='r')
    plt.plot(indices, normalized_positions[:, 1], label=f'Normalized Y (min: {min_y:.2f}, max: {max_y:.2f})', color='g')
    plt.plot(indices, normalized_positions[:, 2], label=f'Normalized Z (min: {min_z:.2f}, max: {max_z:.2f})', color='b')
    plt.title("Normalized Position Components Over Time", fontsize=14)
    plt.xlabel("Index")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/velocity/normalized_positions.png")

    # Non-Normalized Plot: X, Y, Z components over time
    plt.figure(figsize=(10, 8))
    plt.plot(indices, positions[:, 0], label='X Position', color='r')
    plt.plot(indices, positions[:, 1], label='Y Position', color='g')
    plt.plot(indices, positions[:, 2], label='Z Position', color='b')
    plt.title("Position Components Over Time", fontsize=14)
    plt.xlabel("Index")
    plt.ylabel("Position Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/velocity/positions.png")

    # Print mean velocity
    print("Mean velocity:", velocity_magnitude.mean())

    plt.close()



def plot_laser_offset(fov, z_offsets, alpha_offset):
    import matplotlib.pyplot as plt


    num_lines = len(alpha_offset) 
    padded_alpha_offset = list(alpha_offset)

    # Split lines into two halves
    half_num_lines = num_lines // 2

    lower_fov_up  = fov[0] - fov[1]
    lower_fov_down = fov[2] - fov[3]

    # Generate angles for each half between fov limits
    angles_first_half = np.linspace(fov[0], lower_fov_up, half_num_lines)
    angles_second_half = np.linspace(fov[2], lower_fov_down, half_num_lines)


    # Apply laser_offsets to angles
    angles_first_half += np.array(padded_alpha_offset[:half_num_lines])
    angles_second_half += np.array(padded_alpha_offset[half_num_lines:])

    #convert to radians
    angles_first_half = np.radians(angles_first_half)
    angles_second_half = np.radians(angles_second_half)


    # Plotting
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect('equal')

    #Plot angles as direction rays using laser offsets as height coordinate for offset 
    #get x and y coordinates for each angle
    x_first_half = np.cos(angles_first_half)
    y_first_half = np.sin(angles_first_half)
    x_second_half = np.cos(angles_second_half)
    y_second_half = np.sin(angles_second_half)

    #use length 10
    x_first_half *= 10
    y_first_half *= 10
    x_second_half *= 10
    y_second_half *= 10

    up_shift = -z_offsets[0]
    down_shift = -z_offsets[1]

    #add z_offsets to y coordinates
    y_first_half += up_shift
    y_second_half += down_shift

    
    for i in range(half_num_lines):
        plt.plot([0, x_first_half[i]], [up_shift, y_first_half[i]], color='b')
        plt.plot([0, x_second_half[i]], [down_shift, y_second_half[i]], color='r')

   
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Laser Offsets')
    #write file and close
    plt.savefig("plots/laser_offsets.png")


    # Second plot: Y-axis represents indices, X-axis represents offsets
    plt.figure(figsize=(8, 8))
    for line_index in range(num_lines):
        #plot from x = -4 to 4, y = line_index
        angle_index = angles_first_half[line_index] if line_index < half_num_lines else angles_second_half[line_index - half_num_lines]
        delta_fov = fov[1] if line_index < half_num_lines else fov[3]
        #delta fov to radians
        delta_fov = np.radians(delta_fov)
        
        #let angle_index start at 0 so substract fov[0] in radians
        fov_upper = fov[0] if line_index < half_num_lines else fov[2]
        angle_index -= np.radians(fov_upper)
        #divide by delta fov to get normalized value
        angle_index /= -delta_fov
        angle_index*=num_lines/2-1

        #add num_lines/2 if in second half
        if line_index >= half_num_lines:
            angle_index += num_lines/2
        

        plt.plot([0, 1], [line_index, angle_index], color='b' if line_index < half_num_lines else 'r')

    #set y to be decreasing from top to bottom
    plt.gca().invert_yaxis()
    plt.xlabel('Alpha Offset')
    plt.ylabel('Line Index')
    plt.title('Alpha Offsets')
    plt.grid(True)
    plt.savefig("plots/alpha_offsets.png")

    plt.close()

def plot_all(data, channel, smoothing_window=5, descriptor_offset = 0):
    x_values = range(64)  # Ray indices [0, 1, ..., 63]
    #if channel is 0 substract 1 from y_values
    # Plot the data
    y_values = [data[i, channel].clone().detach().cpu().numpy() for i in x_values]
    #if channel == 0:
    #    y_values = [y-1 for y in y_values]

    smoothed_y_values = np.convolve(y_values, 
                    np.ones(smoothing_window)/smoothing_window, 
                    mode='valid')  # Moving average
    valid_x_values = range(smoothing_window//2, 64 - smoothing_window//2)
    plt.plot(x_values, y_values, marker='o', label=f"Channel {channel}")
    plt.plot(valid_x_values, smoothed_y_values, color='red', label=f"Smoothed Channel {channel}")
    #plot smoothed in red

    #write to file
    #axes
    plt.xlabel('ray')
    #get current axis from description
    plt.ylabel(descriptors[channel+descriptor_offset])
    #make folder if not exist
    os.makedirs("raystrength", exist_ok=True)

    #plt.ylim(-0.3, 0.3)
    #save to file
    plt.savefig(f"raystrength/{descriptors[channel+descriptor_offset]}.png")
 
    plt.clf()
    #close plot
    plt.close()

def print_all(channel):
    print(descriptors[channel])
    for i in range(64):
        print(str(i)+ ":",round(self.laser_strength[i, channel].item(), 2), end = ' ')
    print()


    #print('#######################################################################################')
#r


def plot_direction(poses, directions, scale):
    base_rotations = poses[:, :3, :3]

    # Normalize the base rotation matrices (optional: make sure they are valid rotations)
    base_rotations = base_rotations / np.linalg.norm(base_rotations, axis=2, keepdims=True)

    # Compute the rotation matrices for each axis-angle vector (batch processing)
    R_axis_angles = axis_angle_vector_to_rotation_matrix(directions)

    # Apply the axis-angle rotations to the base rotations (matrix multiplication)
    final_rotations = np.matmul(R_axis_angles, base_rotations)

    # Create the plot
    plt.figure(figsize=(10, 8))


    #2d plot for x and y components of forward direction
    forward = final_rotations[:, 1, 0:3]
    forward_base = base_rotations[:, 1, 0:3]
    #plot x component for each index scatter
    plt.scatter(range(forward.shape[0]), forward[:, 0], label='X Forward', color='r')
    plt.scatter(range(forward.shape[0]), forward_base[:, 0], label='X Forward Base', color='r', alpha=0.5)

    #plot y component for each index scatter
    plt.scatter(range(forward.shape[0]), forward[:, 1], label='Y Forward', color='g')
    plt.scatter(range(forward.shape[0]), forward_base[:, 1], label='Y Forward Base', color='g', alpha=0.5)

    #plot z component for each index scatter
    plt.scatter(range(forward.shape[0]), forward[:, 2], label='Z Forward', color='b')
    plt.scatter(range(forward.shape[0]), forward_base[:, 2], label='Z Forward Base', color='b', alpha=0.5)

    # Set labels and title
    plt.xlabel('Index')
    plt.ylabel('Angle')
    plt.title('Forward Direction (XY Components)')
    plt.legend()

    #write to file
    plt.savefig("plots/rotation.png")
    plt.clf()
    # Plot X component
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(forward)), forward[:, 0], label='X Forward', color='r')
    plt.scatter(range(len(forward_base)), forward_base[:, 0], label='X Forward Base', color='y', alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('X Component of Forward Direction')
    plt.legend()
    plt.savefig("plots/rotation_x.png")
    plt.clf()

    # Plot Y component
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(forward)), forward[:, 1], label='Y Forward', color='g')
    plt.scatter(range(len(forward_base)), forward_base[:, 1], label='Y Forward Base', color='c', alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Y Component of Forward Direction')
    plt.legend()
    plt.savefig("plots/rotation_y.png")
    plt.clf()

    # Plot Z component
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(forward)), forward[:, 2], label='Z Forward', color='b')
    plt.scatter(range(len(forward_base)), forward_base[:, 2], label='Z Forward Base', color='purple', alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Z Component of Forward Direction')
    plt.legend()
    plt.savefig("plots/rotation_z.png")
    plt.clf()
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the comparison: Plot original and final rotation in different colors
    # For simplicity, plot the first row of the matrices (can be generalized)
    #ax.quiver(0, 0, 0, base_rotations[:, 0, 0], base_rotations[:, 0, 1], base_rotations[:, 0, 2], color='r', alpha=0.5, label="Base Rotation")
    #ax.quiver(0, 0, 0, final_rotations[:, 0, 0], final_rotations[:, 0, 1], final_rotations[:, 0, 2], color='b', alpha=0.7, label="Final Rotation")

    # Set plot limits for better visualization
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Base Rotation vs Final Rotation (After Axis-Angle)')

    # Add legend
    ax.legend()
    '''
    plt.close()


def plot_trajectory(poses, positions, scale):
    """
    Plot the trajectory of the vehicle in 3D space.

    Parameters:
    poses (np.ndarray): Poses of the vehicle at each time step.
    positions (np.ndarray): Positions of the vehicle at each time step.
    scale (float): Scale factor to adjust positions.
    """
    # Rescale positions
    pose_positions = poses[:, :3, 3]

    positions_off = positions + pose_positions
    positions = pose_positions / scale
    positions_off = positions_off / scale


    # Plot the trajectory in 2d with connected points
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(positions[:, 0], positions[:, 1], marker='o', markersize=0.5, linestyle='-', linewidth=0.4, label='Trajectory')
    #add second trajectory
    plt.plot(positions_off[:, 0], positions_off[:, 1], marker='o', markersize=0.5, linestyle='-', linewidth=0.4, label='Trajectory with offset')
    plt.title("Vehicle Trajectory (2D)", fontsize=14)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("plots/trajectory_2d.png")
    #close
    plt.clf()
    plt.close()


class Trainer(object):
    def __init__(
        self,
        name,   # name of this experiment
        opt,    # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        lidar_metrics=[], # metrics for evaluation, if None, use val_loss to measure performance.
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,   # whether to mute all print
        fp16=False,   # amp optimize level
        eval_interval=50,  # eval once every $ epoch
        max_keep_ckpt=1,   # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",   # the smaller/larger result, the better
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,    # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
        laser_strength = None,
        fov_lidar = None,
        z_offsets = None,
        laser_offsets = None,
        velocity = None,
        R = None,
        T = None,


    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.lidar_metrics = lidar_metrics
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:0" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()
        print("setting laser strength and offsets")
        self.laser_strength = laser_strength
        self.laser_offsets = laser_offsets
        self.velocity = velocity
        self.fov_lidar = fov_lidar
        self.z_offsets = z_offsets
        self.laser_offsets = laser_offsets
        self.R = R
        self.T = T


        model.to(self.device)
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.bce_fn = torch.nn.BCELoss()
        self.cham_fn = chamfer_3DDist()

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        if self.workspace is not None:
            if self.use_checkpoint == "lidar":
                self.log("[INFO] Only loading optimized LiDAR parameters ...")
                self.load_checkpoint(lidar_only=True)
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
        
        self.runtime_train = []
        self.runtime_test = []

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if not self.mute:
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    ### ------------------------------
    @torch.cuda.amp.autocast(enabled=False)
    def load_sensor_settings(self, data, rays_o_lidar, rays_d_lidar, motion=False):
        row_inds = data["row_inds"]  # [B, N]
        col_inds = data["col_inds"]  # [B, N]
        laser_strength = self.laser_strength[row_inds,:]

        poses_lidar = data["poses_lidar"]  # [B, 4, 4]
        poses_before = data["poses_before"]  # [B, 4, 4]
        poses_after = data["poses_after"]  # [B, 4, 4]

        base_translation = poses_lidar[:, :3, 3]  # [B, 3]
        

        interpolated_poses = interpolate_poses(poses_lidar, poses_before, poses_after, num_steps=1024)  # [B, 1024, 4, 4]



        interpolated_poses = interpolated_poses[:, col_inds[0]]  # Shape: [1024, 4, 4]




        translations = interpolated_poses[:, :, :3, 3]  # [B, 1024, 3]

        translation_offset = translations - base_translation# [B, 1024, 3]
        
        
        base_rotation = poses_lidar[:, :3, :3]  # [B, 3, 3]
        rotation = interpolated_poses[:, :, :3, :3]  # [B, 1024, 3, 3]

        #get offset rotation matrix
        #to get a rotation relative to base rotation we need to multiply the inverse of the base rotation
        #with the interpolated rotation
        relative_rotation = torch.matmul(base_rotation.unsqueeze(1).inverse(), rotation)  # [B, 1024, 3, 3]

        #self.plot_poses_arrows(interpolated_poses[0], scale=1.0)  
        return translation_offset, relative_rotation, laser_strength

    def offsets_from_positions(self, position, position_before, position_after, col_inds):
        # Assuming position, position_before, position_after are tensors of shape [B, 3]

        # Compute the distance traveled in forward and backward directions
        distance_traveled_forward = position_after - position  # [B, 3]
        distance_traveled_back = position_before - position  # [B, 3]

        # Define column-related constants
        NUM_COLUMNS = 1024
        HALF_COLUMNS = NUM_COLUMNS // 2

        # Compute distance offsets for the forward direction
        distance_offsets_forward = distance_traveled_forward.unsqueeze(1) / NUM_COLUMNS  # [B, 1, 3]
        half_offsets_forward = torch.arange(1, HALF_COLUMNS + 1, device=position.device).float().unsqueeze(0).unsqueeze(-1)  # [1, HALF_COLUMNS, 1]
        distance_offsets_forward = half_offsets_forward * distance_offsets_forward  # [B, HALF_COLUMNS, 3]

        # Compute distance offsets for the backward direction
        distance_offsets_back = distance_traveled_back.unsqueeze(1) / NUM_COLUMNS  # [B, 1, 3]
        half_offsets_back = torch.arange(HALF_COLUMNS, 0, -1, device=position.device).float().unsqueeze(0).unsqueeze(-1)  # [1, HALF_COLUMNS, 1]
        distance_offsets_back = half_offsets_back * distance_offsets_back  # [B, HALF_COLUMNS, 3]


        # Combine forward and backward offsets by concatenating them along the second dimension
        distance_offsets = torch.cat([distance_offsets_back, distance_offsets_forward], dim=1)  # [B, NUM_COLUMNS, 3]

  


        distance_offsets= distance_offsets[0, col_inds]  # Shape: [128, 3]


        return distance_offsets
        






    def offsets_from_velocities(self, velocity, col_inds, delta_time=0.1):
        # Calculate distance traveled per time step (100ms)
        delta_time = 0.1  
        distance_traveled = velocity * delta_time  # [B, 3]

        # Define column-related constants
        NUM_COLUMNS = 1024
        HALF_COLUMNS = NUM_COLUMNS // 2

        # Compute per-column distance traveled in each dimension
        distance_traveled_per_column = distance_traveled / NUM_COLUMNS  # [B, 3]

        # Precompute offsets for all columns (broadcast to [B, NUM_COLUMNS, 3])
        column_offsets = torch.arange(-HALF_COLUMNS, HALF_COLUMNS, device=distance_traveled.device).float()
        column_offsets = column_offsets[:, None]  # Shape: [NUM_COLUMNS, 1]
        distance_offsets = column_offsets * distance_traveled_per_column.unsqueeze(1)  # [B, NUM_COLUMNS, 3]

    


        # Flatten the batch dimension for easier indexing
        distance_offsets_flat = distance_offsets[0]  # Shape: [1024, 3]
        col_inds_flat = col_inds[0]  # Shape: [1024]

        # Use col_inds to index distance_offsets
        distances_traveled = distance_offsets_flat[col_inds_flat]  # Shape: [1024, 3]

        # If necessary, add the batch dimension back
        distances_traveled = distances_traveled.unsqueeze(0)  # Shape: [1, 1024, 3]
        return distances_traveled


    @torch.cuda.amp.autocast(enabled=False)
    def load_sensor_settings_d(self, data, rays_o_lidar, rays_d_lidar, motion=False):


        #gives line ids for each ray
        row_inds = data["row_inds"]  # [B, N]

        laser_strength = self.laser_strength[row_inds,:]
        if motion:
            col_inds = data["col_inds"]  # [B, N]


            poses_lidar = data["poses_lidar"]  # [B, 4, 4]
            poses_before = data["poses_before"]  # [B, 4, 4]
            poses_after = data["poses_after"]  # [B, 4, 4]

            #laser_offset = self.laser_offsets[row_inds,:]

            #laser_R = Exp(laser_offset[:, :3])

            distances_traveled_global = self.offsets_from_positions(poses_lidar[:, :3, 3], poses_before[:, :3, 3], poses_after[:, :3, 3], col_inds)  # [B, N, 3]



            
            if False:

                #add homogenous coordinates
                rays_o_lidar = torch.cat([rays_o_lidar, torch.ones_like(rays_o_lidar[:, :, :1])], dim=-1)
                inv_pose = torch.inverse(poses_lidar)
                inv_pose_rot = inv_pose[:,:3,:3]
                rays_o_lidar = torch.matmul(inv_pose, rays_o_lidar.unsqueeze(-1)).squeeze(-1)
                rays_o_lidar = rays_o_lidar[:,:,:3]
                #for direction we only need the rotation part of the pose so we do not need homogenous coordinates

                rays_d_lidar = torch.matmul(inv_pose_rot, rays_d_lidar.unsqueeze(-1)).squeeze(-1)

                batch_idx = 0  # Visualize the first batch
                rays_d_batch = rays_d_lidar[batch_idx]  # [N, 3]
                rays_o_batch = rays_o_lidar[batch_idx]  # [N, 3] 
                row_batch = row_inds[batch_idx]  # [N]
                col_batch = col_inds[batch_idx]  # [N]
                inv_pose = inv_pose[batch_idx]  # [4, 4]
                
                rays_o_batch/=self.opt.scale


                #normalize rays_d_batch
                #set z to 0
                rays_d_batch[:, 2] = 0
                rays_d_batch = F.normalize(rays_d_batch, dim=-1)

                        # Extract x, y components of ray directions
                u = rays_d_batch[:, 0]  # x-component of direction
                v = rays_d_batch[:, 1]  # y-component of direction


                x= rays_o_batch[:, 0]
                y= rays_o_batch[:, 1]
                z= rays_o_batch[:, 2]

                        #convert tensors to numpy
                u = u.cpu().detach().numpy()
                v = v.cpu().detach().numpy()
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                z = z.cpu().detach().numpy()



                col_batch = col_batch.cpu().numpy()
                row_batch = row_batch.cpu().numpy()

                #col_batch is from 0 to 1024 use these to color 
                #use colormap from black to white
                normalized_col_batch = col_batch / 1024  # Assuming col_batch ranges from 0 to 1024
                colmap = plt.get_cmap('gray')(normalized_col_batch) 


        
                # Plot the arrows
                plt.figure(figsize=(10, 10))
                #plt.scatter(u, v, c=colmap, label="Ray Directions", s=10) 
                plt.scatter(x+u, y+v, c=colmap, label="Ray Directions", s=10)

                plt.xlim(-2, 2)
                plt.ylim(-2, 2)


        

                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Visualization of Ray Directions (XY Components)")
                plt.grid(True)
                #plt.axis("equal")
                #plt.xlim(0, np.max(col_batch))  # Limit x-axis to max column range
                #plt.ylim(0, 64)  # Limit y-axis to 0-64 range

                #write to file
                plt.savefig("plots/ray_directions.png")

                #close plot
                plt.close()

                rays_o_lidar = torch.cat([rays_o_lidar, torch.ones_like(rays_o_lidar[:, :, :1])], dim=-1)
                rays_o_lidar = torch.matmul(poses_lidar, rays_o_lidar.unsqueeze(-1)).squeeze(-1)
                rays_o_lidar = rays_o_lidar[:,:,:3]
                #without this it gets cast to float16
                #but why?
                rot_pose  = poses_lidar[:,:3,:3]
                rays_d_lidar = torch.matmul(rot_pose, rays_d_lidar.unsqueeze(-1)).squeeze(-1)


        else:
            #distance is 0s
            distances_traveled_global = torch.zeros_like(rays_o_lidar)
        return distances_traveled_global, laser_strength
    

   

   


    def plot_poses_arrows(poses, scale=1.0):

        #copy poses to cpu 
        poses = poses.cpu().detach().numpy()

    
        # Extract positions and rotations from poses
        positions = poses[:, :3, 3] / scale
        rotations = poses[:, :3, :3]

        #get angles from rotation matrix
        angles = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])

        #get x and y angles
        x = np.cos(angles)
        y = np.sin(angles)

        
        
        # Create plot
        plt.figure(figsize=(10, 10))
        
        #plot x and y positions and arrow for x and y rotation
        plt.scatter(positions[:, 0], positions[:, 1], label='Positions', color='b')
        plt.quiver(positions[:, 0], positions[:, 1], x, y, color='r', scale=30, width=0.005, headwidth=4, headlength=5, label='Rotations')
        

        # Set equal aspect ratio
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Vehicle Poses')
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/poses_arrows.png')
        plt.close()

        return 0


    

    def train_step(self, data):
        # Initialize all returned values
        pred_intensity = None
        gt_intensity = None
        pred_depth = None
        gt_depth = None
        loss = 0

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time'] # [B, 1]
        images_lidar = data["images_lidar"]  # [B, N, 3]

        motion_offset, motion_rotation, laser_strength = self.load_sensor_settings(data, rays_o_lidar, rays_d_lidar)


        #self.plot_interpolation(rays_o_lidar, rays_d_lidar, data, motion_offset, motion_rotation)
        gt_raydrop = images_lidar[:, :, 0]
        gt_intensity = images_lidar[:, :, 1] * gt_raydrop
        gt_depth = images_lidar[:, :, 2] * gt_raydrop

        outputs_lidar = self.model.render(
            rays_o_lidar + motion_offset,
            torch.matmul(motion_rotation, rays_d_lidar.unsqueeze(-1)).squeeze(-1),
            time_lidar,
            staged=False,
            perturb=True,
            force_all_rays=False if self.opt.patch_size_lidar == 1 else True,
            **vars(self.opt),
        )
        

        pred_raydrop = outputs_lidar["image_lidar"][:, :, 0]
        pred_intensity = (outputs_lidar["image_lidar"][:, :, 1] * laser_strength[:, :, 0]) * gt_raydrop 
        pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop

 

        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)

        # label smoothing for ray-drop
        smooth = self.opt.smooth_factor # 0.2
        gt_raydrop_smooth = gt_raydrop.clamp(smooth, 1-smooth)

        #offset loss is difference between offsets and 0.075
        #offset_loss = torch.abs(torch.abs(self.z_offsets[0]-self.z_offsets[1]) - 0.075).mean()

        #strength loss that punishes values above 1 
        strength_loss = torch.relu(laser_strength - 1.0)

        lidar_loss = (
            self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth)
            + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop_smooth)
            + self.opt.alpha_i * self.criterion["intensity"](pred_intensity, gt_intensity)
            +  0.01*strength_loss.mean()
         #   + 0.001*offset_loss

        )
        pred_intensity = pred_intensity.unsqueeze(-1)
        gt_intensity = gt_intensity.unsqueeze(-1)

        # main loss
        loss = lidar_loss.sum()

        # additional CD Loss
        pred_lidar = rays_d_lidar * pred_depth.unsqueeze(-1) / self.opt.scale
        gt_lidar = rays_d_lidar * gt_depth.unsqueeze(-1) / self.opt.scale
        dist1, dist2, _, _ = self.cham_fn(pred_lidar, gt_lidar)
        chamfer_loss = (dist1 + dist2).mean() * 0.5
        loss = loss + chamfer_loss

        if self.opt.flow_loss:
            frame_idx = int(time_lidar * (self.opt.num_frames - 1))
            pc = self.pc_list[f"{frame_idx}"]
            pc = torch.from_numpy(pc).cuda().float().contiguous()

            pred_flow = self.model.flow(pc, time_lidar)
            pred_flow_forward = pred_flow["forward"]
            pred_flow_backward = pred_flow["backward"]

            # two-step consistency
            for step in [1, 2]:
                if f"{frame_idx+step}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_forward * step
                    pc_forward = self.pc_list[f"{frame_idx+step}"]
                    pc_forward = torch.from_numpy(pc_forward).cuda().float().contiguous()
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_forward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5
                    loss = loss + chamfer_dist

                if f"{frame_idx-step}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_backward * step
                    pc_backward = self.pc_list[f"{frame_idx-step}"]
                    pc_backward = torch.from_numpy(pc_backward).cuda().float().contiguous()
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_backward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5
                    loss = loss + chamfer_dist

            # regularize flow on the ground
            ground = self.pc_ground_list[f"{frame_idx}"]
            ground = torch.from_numpy(ground).cuda().float().contiguous()
            zero_flow = self.model.flow(ground, torch.rand(1).to(time_lidar))
            loss = loss + 0.001 * (zero_flow["forward"].abs().sum() + zero_flow["backward"].abs().sum())

        # line-of-sight loss
        if self.opt.urf_loss:
            eps = 0.02 * 0.1 ** min(self.global_step / self.opt.iters, 1)
            # gt_depth [B, N]
            weights = outputs_lidar["weights"] # [B*N, T]
            z_vals = outputs_lidar["z_vals"]

            depth_mask = gt_depth.reshape(z_vals.shape[0], 1) > 0.0
            mask_empty = (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) - eps)) | (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) + eps))
            loss_empty = ((mask_empty * weights) ** 2).sum() / depth_mask.sum()

            loss = loss + 0.1 * loss_empty

            mask_near = (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) - eps)) & (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) + eps))
            distance = mask_near * (z_vals - gt_depth.reshape(z_vals.shape[0], 1))
            sigma = eps / 3.
            distr = 1.0 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-(distance ** 2 / (2 * sigma ** 2)))
            distr /= distr.max()
            distr *= mask_near
            loss_near = ((mask_near * weights - distr) ** 2).sum() / depth_mask.sum()

            loss = loss + 0.1 * loss_near

        # gradient loss
        if isinstance(self.opt.patch_size_lidar, int):
            patch_size_x, patch_size_y = self.opt.patch_size_lidar, self.opt.patch_size_lidar
        elif len(self.opt.patch_size_lidar) == 1:
            patch_size_x, patch_size_y = self.opt.patch_size_lidar[0], self.opt.patch_size_lidar[0]
        else:
            patch_size_x, patch_size_y = self.opt.patch_size_lidar

        if patch_size_x > 1:
            pred_depth = pred_depth.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous() / self.opt.scale
            if self.opt.sobel_grad:
                pred_grad_x = F.conv2d(
                    pred_depth,
                    torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                    padding=1,
                    )
                pred_grad_y = F.conv2d(
                    pred_depth,
                    torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                    padding=1,
                    )
            else:
                pred_grad_y = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
                pred_grad_x = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])

            dy = torch.abs(pred_grad_y)
            dx = torch.abs(pred_grad_x)

            if self.opt.grad_norm_smooth:
                grad_norm = torch.mean(torch.exp(-dx)) + torch.mean(torch.exp(-dy))
                # print('grad_norm', grad_norm)
                loss = loss + self.opt.alpha_grad_norm * grad_norm

            if self.opt.spatial_smooth:
                spatial_loss = torch.mean(dx**2) + torch.mean(dy**2)
                # print('spatial_loss', spatial_loss)
                loss = loss + self.opt.alpha_spatial * spatial_loss

            if self.opt.tv_loss:
                tv_loss = torch.mean(dx) + torch.mean(dy)
                # print('tv_loss', tv_loss)
                loss = loss + self.opt.alpha_tv * tv_loss

            if self.opt.grad_loss:
                gt_depth = gt_depth.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous() / self.opt.scale
                gt_raydrop = gt_raydrop.view(-1, patch_size_x, patch_size_y, 1).permute(0, 3, 1, 2).contiguous()

                # sobel
                if self.opt.sobel_grad:
                    gt_grad_y = F.conv2d(
                        gt_depth,
                        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                        padding=1,
                        )

                    gt_grad_x = F.conv2d(
                        gt_depth,
                        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
                        padding=1,
                        )
                else:
                    gt_grad_y = gt_depth[:, :, :-1, :] - gt_depth[:, :, 1:, :]
                    gt_grad_x = gt_depth[:, :, :, :-1] - gt_depth[:, :, :, 1:]

                grad_clip_x = 0.01
                grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip_x, 1, 0)
                grad_clip_y = 0.01
                grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip_y, 1, 0)
                if self.opt.sobel_grad:
                    mask_dx = gt_raydrop * grad_mask_x
                    mask_dy = gt_raydrop * grad_mask_y
                else:
                    mask_dx = gt_raydrop[:, :, :, :-1] * grad_mask_x
                    mask_dy = gt_raydrop[:, :, :-1, :] * grad_mask_y

                if self.opt.depth_grad_loss == "cos":
                    patch_num = pred_grad_x.shape[0]
                    grad_loss = self.criterion["grad"](
                        (pred_grad_x * mask_dx).reshape(patch_num, -1),
                        (gt_grad_x * mask_dx).reshape(patch_num, -1),
                    )
                    grad_loss = 1 - grad_loss
                else:
                    grad_loss = self.criterion["grad"](
                        pred_grad_x * mask_dx, 
                        gt_grad_x * mask_dx
                    )
                loss = loss + self.opt.alpha_grad * grad_loss.sum()

        return (
            pred_intensity,
            gt_intensity,
            pred_depth,
            gt_depth,
            loss,
        )

    def eval_step(self, data):
        pred_intensity = None
        pred_depth = None
        pred_raydrop = None
        gt_intensity = None
        gt_depth = None
        gt_raydrop = None
        loss = 0

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time']
        images_lidar = data["images_lidar"]  # [B, H, W, 3]
        H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
        #gives line ids for each ray
        motion_offset, motion_rotation, optimized_laserstrength = self.load_sensor_settings(data, rays_o_lidar, rays_d_lidar)

       

        gt_raydrop = images_lidar[:, :, :, 0]
        gt_intensity = images_lidar[:, :, :, 1] * gt_raydrop
        gt_depth = images_lidar[:, :, :, 2] * gt_raydrop

        outputs_lidar = self.model.render(
            rays_o_lidar + motion_offset,
            torch.matmul(motion_rotation, rays_d_lidar.unsqueeze(-1)).squeeze(-1),
            time_lidar,
            staged=True,
            perturb=False,
            **vars(self.opt),
        )

        pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
        mult_laser = optimized_laserstrength[:,:,0].reshape(-1, H_lidar, W_lidar)
        add_laser = optimized_laserstrength[:,:,1].reshape(-1, H_lidar, W_lidar)


        pred_intensity = pred_intensity *mult_laser+add_laser
        

        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)
        if self.use_refine:
            pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
            pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)

        
        lidar_loss = (
            self.opt.alpha_d * self.criterion["depth"](pred_depth * raydrop_mask, gt_depth).mean()
            + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
            + self.opt.alpha_i * self.criterion["intensity"]((pred_intensity)* raydrop_mask, gt_intensity).mean()
        )

        loss = lidar_loss
        
        return (
            pred_intensity,
            pred_depth,
            pred_raydrop,
            gt_intensity,
            gt_depth,
            gt_raydrop,
            loss,
        )

    def test_step(self, data, perturb=False):
        pred_raydrop = None
        pred_intensity = None
        pred_depth = None

        rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]

        rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
        time_lidar = data['time']
        H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]


        motion_offset, motion_rotation, optimized_laserstrength = self.load_sensor_settings(data, rays_o_lidar, rays_d_lidar)
        
        outputs_lidar = self.model.render(
            rays_o_lidar + motion_offset,
            torch.matmul(motion_rotation, rays_d_lidar.unsqueeze(-1)).squeeze(-1),
            time_lidar,
            staged=True,
            perturb=perturb,
            **vars(self.opt),
        )

        pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
        pred_raydrop = pred_rgb_lidar[:, :, :, 0]
        pred_intensity = pred_rgb_lidar[:, :, :, 1]
        pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
        mult_laser = optimized_laserstrength[:,:,0].reshape(-1, H_lidar, W_lidar)
        add_laser = optimized_laserstrength[:,:,1].reshape(-1, H_lidar, W_lidar)

        if self.opt.raydrop_loss == 'bce':
            pred_raydrop = F.sigmoid(pred_raydrop)
        if self.use_refine:
            pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
            pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
        raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
        if self.opt.alpha_r > 0:
            pred_intensity = (pred_intensity * mult_laser + add_laser)  * raydrop_mask 
            pred_depth = pred_depth * raydrop_mask

        return pred_raydrop, pred_intensity, pred_depth

    ### ------------------------------

    def train_one_epoch(self, loader):
        log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log(
            f"[{log_time}] ==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )
        #log fov_lidar and z_offsets
        self.log(f"fov_lidar: {self.fov_lidar.detach().cpu().numpy()}")
        self.log(f"z_offsets: {self.z_offsets.detach().cpu().numpy()}")

        if self.laser_strength.grad is not None:
            #print("mean grad", self.laser_strength.grad.mean().item())
            #for i in range(self.laser_strength.shape[1]):
            #    print_all(i)
            for i in range(self.laser_strength.shape[1]):
                plot_all(self.laser_strength,i)

        plot_laser_offset(self.fov_lidar.detach().cpu().numpy(), self.z_offsets.detach().cpu().numpy(), self.laser_offsets.detach().cpu().numpy())

        plot_trajectory(loader._data.poses_lidar.clone().detach().cpu().numpy(), loader._data.T.clone().detach().cpu().numpy(), self.opt.scale)

        plot_direction(loader._data.poses_lidar.clone().detach().cpu().numpy(), loader._data.R.clone().detach().cpu().numpy(), self.opt.scale)

        plt.close('all')
        total_loss = 0

        self.model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                (
                    pred_intensity,
                    gt_intensity,
                    pred_depth,
                    gt_depth,
                    loss,
                ) = self.train_step(data)

 
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )

            if self.scheduler_update_every_step:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                )
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        self.log(f"average_loss: {average_loss}.")



        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
        pbar.close()

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        for metric in self.lidar_metrics:
            metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    (
                        preds_intensity,
                        preds_depth,
                        preds_raydrop,
                        gt_intensity,
                        gt_depth,
                        gt_raydrop,
                        loss,
                    ) = self.eval_step(data)

                preds_mask = torch.where(preds_raydrop > 0.5, 1, 0)

                loss_val = loss.item()
                total_loss += loss_val

                for i, metric in enumerate(self.lidar_metrics):
                    if i == 0:  # hard code
                        metric.update(preds_raydrop, gt_raydrop)
                    elif i == 1:
                        metric.update(preds_intensity*preds_mask, gt_intensity)
                    else:
                        metric.update(preds_depth*preds_mask, gt_depth)

                save_path_pred = os.path.join(
                    self.workspace,
                    "validation",
                    f"{name}_{self.local_step:04d}.png",
                )
                os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)
                

                pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                img_raydrop = (pred_raydrop * 255).astype(np.uint8)
                img_raydrop = cv2.cvtColor(img_raydrop, cv2.COLOR_GRAY2BGR)



                pred_intensity = preds_intensity[0].detach().cpu().numpy()
                #clip intensity to 0-1
                #pred_intensity = np.clip(pred_intensity, 0, 1)
                img_intensity = (pred_intensity * 255).astype(np.uint8)
                img_intensity = cv2.applyColorMap(img_intensity, 1)
                
                pred_depth = preds_depth[0].detach().cpu().numpy()
                img_depth = (pred_depth * 255).astype(np.uint8)
                img_depth = cv2.applyColorMap(img_depth, 20)

                preds_mask = preds_mask[0].detach().cpu().numpy()
                img_mask = (preds_mask * 255).astype(np.uint8)
                img_raydrop_masked = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

                img_intensity_masked = (pred_intensity * preds_mask * 255).astype(np.uint8)
                img_intensity_masked = cv2.applyColorMap(img_intensity_masked, 1)
                
                img_depth_masked = (pred_depth * preds_mask * 255).astype(np.uint8)
                img_depth_masked = cv2.applyColorMap(img_depth_masked, 20)

                #img_pred = cv2.vconcat([img_raydrop, img_intensity, img_depth, 
                #                        img_raydrop_masked, img_intensity_masked, #img_depth_masked])

                img_raydrop_gt = (gt_raydrop[0].detach().cpu().numpy() * 255).astype(np.uint8)
                img_raydrop_gt = cv2.cvtColor(img_raydrop_gt, cv2.COLOR_GRAY2BGR)
                img_intensity_gt = (gt_intensity[0].detach().cpu().numpy() * 255).astype(np.uint8)
                img_intensity_gt = cv2.applyColorMap(img_intensity_gt, 1)
                img_depth_gt = (gt_depth[0].detach().cpu().numpy() * 255).astype(np.uint8)
                img_depth_gt = cv2.applyColorMap(img_depth_gt, 20)

                img_gt = cv2.vconcat([img_raydrop, img_intensity, img_depth,
                                      img_raydrop_gt, img_intensity_gt, img_depth_gt])

                
                cv2.imwrite(save_path_pred, img_gt)

        
                ## save point clouds
                # pred_lidar = pano_to_lidar(
                #     pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                # )
                # np.save(
                #     os.path.join(
                #         self.workspace,
                #         "validation",
                #         f"{name}_{self.local_step:04d}_lidar.npy",
                #     ),
                #     pred_lidar,
                # )

                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                )
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        if len(self.lidar_metrics) > 0:
            result = self.lidar_metrics[-1].measure()[0]  # hard code
            self.stats["results"].append(
                result if self.best_mode == "min" else -result
            )  # if max mode, use -result
        else:
            self.stats["results"].append(
                average_loss
            )  # if no metric, choose best by min loss

        np.set_printoptions(linewidth=150, suppress=True, precision=8)
        for i, metric in enumerate(self.lidar_metrics):
            if i == 1:
                self.log(f"== ↓ Final pred ↓ == RMSE{' '*6}MedAE{' '*6}LPIPS{' '*8}SSIM{' '*8}PSNR ===")
            self.log(metric.report(), style="blue")
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate")
            metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, refine_loader, max_epochs):
        if self.use_tensorboardX:
            summary_path = os.path.join(self.workspace, "run", self.name)
            self.writer = tensorboardX.SummaryWriter(summary_path)

        print("flow_loss", self.opt.flow_loss)
        if self.opt.flow_loss:
            self.process_pointcloud(refine_loader)

        change_dataloder = False
        if self.opt.change_patch_size_lidar[0] > 1:
            change_dataloder = True
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            if change_dataloder:
                if self.epoch % self.opt.change_patch_size_epoch == 0:
                    train_loader._data.patch_size_lidar = self.opt.change_patch_size_lidar
                    self.opt.patch_size_lidar = self.opt.change_patch_size_lidar
                else:
                    train_loader._data.patch_size_lidar = 1
                    self.opt.patch_size_lidar = 1

            self.train_one_epoch(train_loader)

            if self.workspace is not None:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.use_refine = False
                self.evaluate_one_epoch(valid_loader)

        #print laser_strength

        #self.refine(refine_loader)

        if self.use_tensorboardX:
            self.writer.close()

    def evaluate(self, loader, name=None, refine=True):
        self.use_refine = refine
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, refine=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        if write_video:
            all_preds = []
            all_preds_depth = []

        self.use_refine = refine

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds_raydrop, preds_intensity, preds_depth = self.test_step(data)

                pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                    loader._data.H_lidar, loader._data.W_lidar
                )
                pred_raydrop = (pred_raydrop * 255).astype(np.uint8)

                pred_intensity = preds_intensity[0].detach().cpu().numpy()
                pred_intensity = (pred_intensity * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()


                lidar_K = loader._data.intrinsics_lidar.cpu().detach().numpy()
                z_offsets = self.opt.z_offsets.cpu().detach().numpy()
                laser_offsets = self.opt.laser_offsets.cpu().detach().numpy()


                pred_lidar = pano_to_lidar(
                    pred_depth / self.opt.scale, lidar_K, z_offsets, laser_offsets
                )

                np.save(
                    os.path.join(save_path, f"test_{name}_{i+1:04d}_depth_lidar.npy"),
                    pred_lidar,
                )

                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(cv2.cvtColor(cv2.applyColorMap(pred_intensity, 1), cv2.COLOR_BGR2RGB))
                    all_preds_depth.append(cv2.cvtColor(cv2.applyColorMap(pred_depth, 20), cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(
                        os.path.join(save_path, f"test_{name}_{i+1:04d}_raydrop.png"),
                        pred_raydrop,
                    )
                    cv2.imwrite(
                        os.path.join(
                            save_path, f"test_{name}_{i+1:04d}_intensity.png"
                        ),
                        cv2.applyColorMap(pred_intensity, 1),
                    )
                    cv2.imwrite(
                        os.path.join(save_path, f"test_{name}_{i+1:04d}_depth.png"),
                        cv2.applyColorMap(pred_depth, 20),
                    )

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_lidar_rgb.mp4"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_depth.mp4"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )

        self.log(f"==> Finished Test.")


    def plot_interpolation(self, torch_rays_o, torch_rays_d, data, interpolated_translations, interpolated_rotations):
            

            col_inds = data["col_inds"].cpu().numpy()[0]  # [B, N]

            poses_lidar = data["poses_lidar"]  # [B, 4, 4]
            poses_before = data["poses_before"]  # [B, 4, 4]
            poses_after = data["poses_after"]  # [B, 4, 4]


            prev_trans = poses_before[:, :3, 3].detach().cpu().numpy()[0]
            trans = poses_lidar[:, :3, 3].detach().cpu().numpy()[0]
            next_trans = poses_after[:, :3, 3].detach().cpu().numpy()[0]

            interpolated_forwards = interpolated_rotations[:, :3, 1].detach().cpu().numpy()[0]
         
            prev_forward = poses_before[:, :3, 1].detach().cpu().numpy()[0]
            forward = poses_lidar[:, :3, 1].detach().cpu().numpy()[0]
            next_forward = poses_after[:, :3, 1].detach().cpu().numpy()[0]


            rays_o = torch_rays_o.detach().cpu().numpy()[0] + interpolated_translations.detach().cpu().numpy()[0]

            rays_d = torch.matmul(interpolated_rotations, torch_rays_d.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()[0]

            rays_d = rays_d[:, :2]
            #normalize
            rays_d = rays_d / np.linalg.norm(rays_d, axis=1)[:, None]*10
            # Plot 2D rays from rays_o to direction

            print(rays_o[:,0].shape, col_inds.shape, rays_d.shape)
            # Use existing figure, just add new plots
            q = plt.quiver(rays_o[:, 0], rays_o[:, 1],
                    rays_d[:, 0], rays_d[:, 1],
                    col_inds,  # Color by column index
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
            
            #plt.quiver(rays_o[:, 0], rays_o[:, 1], interpolated_forwards[:, 0], interpolated_forwards[:, 1], color='y', scale=10, width=0.001)

            plt.colorbar(q)  # Add colorbar
            plt.axis('equal')
            plt.grid(True)
            #plt.show()
            #save
            plt.savefig("debug_proj/rays.png")
            print("saved to debug_proj/rays.png")
            #clear plot
            plt.clf()



    def refine(self, loader):
        if self.ema is not None:
            self.ema.copy_to() # load ema model weights
            self.ema = None    # no need for final model weights

        self.model.eval()

        raydrop_input_list = []
        raydrop_gt_list = []

   
        self.log("Preparing for Raydrop Refinemet ...")
        for i, data in enumerate(loader):
                    #gives line ids for each ray


            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]


            motion_offset, motion_rotation, optimized_laserstrength = self.load_sensor_settings(data, rays_o_lidar, rays_d_lidar)
  
            time_lidar = data['time']
            H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
            gt_raydrop = data["images_lidar"][:, :, :, 0].unsqueeze(0)

     



            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                with torch.no_grad():
                    outputs_lidar = self.model.render(
                        rays_o_lidar + motion_offset,
                        torch.matmul(motion_rotation, rays_d_lidar.unsqueeze(-1)).squeeze(-1),
                        time_lidar,
                        staged=True,
                        max_ray_batch=4096,
                        perturb=False,
                        **vars(self.opt),
                    )

            

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(-1, H_lidar, W_lidar, 2)
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            pred_intensity = pred_rgb_lidar[:, :, :, 1] * optimized_laserstrength[:,:,0].reshape(-1, H_lidar, W_lidar)+optimized_laserstrength[:,:,1].reshape(-1, H_lidar, W_lidar)
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)

            raydrop_input = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)

            raydrop_input_list.append(raydrop_input)
            raydrop_gt_list.append(gt_raydrop)
            if i % 10 == 0:
                print(f"{i+1}/{len(loader)}")

        torch.cuda.empty_cache()

        raydrop_input = torch.cat(raydrop_input_list, dim=0).cuda().float().contiguous() # [B, 3, H, W]
        raydrop_gt = torch.cat(raydrop_gt_list, dim=0).cuda().float().contiguous()       # [B, 1, H, W]

        self.model.unet.train()

        loss_total = []

        refine_bs = 16 # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
        refine_epoch = 100#0

        optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=0.001, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)

        self.log("Start UNet Optimization ...")
        for i in range(refine_epoch):
            optimizer.zero_grad()

            if refine_bs is not None:
                idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
                input = raydrop_input[idx,...]
                gt = raydrop_gt[idx,...]
            else:
                input = raydrop_input
                gt = raydrop_gt

            # random mask
            mask = torch.ones_like(input).to(input.device)
            box_num_max = 32
            box_size_y_max = int(0.1 * input.shape[2])
            box_size_x_max = int(0.1 * input.shape[3])
            for j in range(np.random.randint(box_num_max)):
                box_size_y = np.random.randint(1, box_size_y_max)
                box_size_x = np.random.randint(1, box_size_x_max)
                yi = np.random.randint(input.shape[2]-box_size_y)
                xi = np.random.randint(input.shape[3]-box_size_x)
                mask[:, :, yi:yi+box_size_y, xi:xi+box_size_x] = 0.
            input = input * mask

            raydrop_refine = self.model.unet(input)
            bce_loss = self.bce_fn(raydrop_refine, gt)
            loss = bce_loss

            loss.backward()

            loss_total.append(loss.item())

            if i % 50 == 0:
                log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.log(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")

            optimizer.step()
            scheduler.step()

        state = {
            "epoch": self.epoch,
            "model": self.model.state_dict()
            }
        file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}_refine.pth"
        torch.save(state, file_path)

        torch.cuda.empty_cache()

    def process_pointcloud(self, loader):
        self.log("Preparing Point Clouds ...")
        self.pc_list = {}
        self.pc_ground_list = {}
        for i, data in enumerate(loader):
            # pano to lidar
            images_lidar = data["images_lidar"]
            gt_raydrop = images_lidar[:, :, :, 0]
            gt_depth = images_lidar[:, :, :, 2] * gt_raydrop
            gt_lidar = pano_to_lidar(
                gt_depth.squeeze(0).clone().detach().cpu().numpy() / self.opt.scale, 
                loader._data.intrinsics_lidar.clone().detach().cpu().numpy(),
                z_offsets = self.opt.z_offsets.clone().detach().cpu().numpy()
            )
            # remove ground
            points, ground = point_removal(gt_lidar)
            # transform
            pose = data["poses_lidar"].squeeze(0)
            pose = pose.clone().detach().cpu().numpy()
            points = points * self.opt.scale
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            points = (pose @ points.T).T[:,:3]
            ground = ground * self.opt.scale
            ground = np.hstack((ground, np.ones((ground.shape[0], 1))))
            ground = (pose @ ground.T).T[:,:3]
            time_lidar = data["time"]
            frame_idx = int(time_lidar * (self.opt.num_frames - 1))
            self.pc_list[f"{frame_idx}"] = points
            self.pc_ground_list[f"{frame_idx}"] = ground
            if i % 10 == 0:
                print(f"{i+1}/{len(loader)}")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
            "laser_strength": self.laser_strength.detach().cpu(),
            "laser_offsets": self.laser_offsets.detach().cpu(),
            "laser_offsets": self.laser_offsets.detach().cpu(),
            "z_offsets": self.z_offsets.detach().cpu(),
            "fov_lidar": self.fov_lidar.detach().cpu(),
            "R": self.R.detach().cpu(),
            "T": self.T.detach().cpu()
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            torch.save(state, file_path)

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False, lidar_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)


        if "laser_strength" in checkpoint_dict:
            self.laser_strength = torch.nn.Parameter(checkpoint_dict['laser_strength'].to(self.device))
            #add laser_strength to optimizer
            self.optimizer.add_param_group({'params': self.laser_strength, 'lr': self.opt.lr *  0.1})
            print("laser_strength loaded")


        if "z_offsets" in checkpoint_dict:
            self.z_offsets = torch.nn.Parameter(checkpoint_dict['z_offsets'].to(self.device))
            self.optimizer.add_param_group({'params': self.z_offsets, 'lr': self.opt.lr *  0.001})

            print("z_offsets loaded")

        if "fov_lidar" in checkpoint_dict:
            self.fov_lidar = torch.nn.Parameter(checkpoint_dict['fov_lidar'].to(self.device))
            print("fov_lidar loaded")
            self.optimizer.add_param_group({'params': self.fov_lidar, 'lr': self.opt.lr *  0.001})

        if "laser_offsets" in checkpoint_dict:
            self.laser_offsets = torch.nn.Parameter(checkpoint_dict['laser_offsets'].to(self.device))
            print("laser_offsets loaded")
            self.optimizer.add_param_group({'params': self.laser_offsets, 'lr': self.opt.lr *  0.001})

        if "velocity" in checkpoint_dict:
            self.velocity = torch.nn.Parameter(checkpoint_dict['velocity'].to(self.device))
            self.optimizer.add_param_group({'params': self.velocity, 'lr': self.opt.lr *  0.001})
            print("velocity loaded")

        if "R" in checkpoint_dict:
            self.R = torch.nn.Parameter(checkpoint_dict['R'].to(self.device))
            self.optimizer.add_param_group({'params': self.R, 'lr': self.opt.lr *  0.1})
            print("R loaded")

        if "T" in checkpoint_dict:
            self.T = torch.nn.Parameter(checkpoint_dict['T'].to(self.device))
            self.optimizer.add_param_group({'params': self.T, 'lr': self.opt.lr *  0.1})
            print("T loaded")

        if lidar_only:
            return

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return


        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if model_only:
            return

        if "stats" in checkpoint_dict:
            self.stats = checkpoint_dict["stats"]
        if "epoch" in checkpoint_dict:
            self.epoch = checkpoint_dict["epoch"]
        if "global_step" in checkpoint_dict:
            self.global_step = checkpoint_dict["global_step"]
            self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")


 


        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
