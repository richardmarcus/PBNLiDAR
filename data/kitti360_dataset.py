import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from data.base_dataset import get_lidar_rays, BaseDataset


def vec2skew(v):
    """
    Convert a batch of vectors to their corresponding skew-symmetric matrices.
    :param v:  (B, 3) or (3,) torch tensor
    :return:   (B, 3, 3) or (3, 3)
    """
    if v.ndim == 1:  # Handle single vector case
        v = v.unsqueeze(0)  # Convert to (1, 3) for uniform processing

    zero = torch.zeros(v.size(0), 1, dtype=torch.float32, device=v.device)  # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], dim=1)  # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], dim=1)  # (B, 3)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], dim=1)  # (B, 3)

    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (B, 3, 3)

    #if skew_v.size(0) == 1:  # If input was a single vector, return (3, 3)
    #    skew_v = skew_v.squeeze(0)
    return skew_v

def Exp(r):
    """
    so(3) vector to SO(3) matrix for a batch of vectors
    :param r: (B, 3) or (3,) axis-angle, torch tensor
    :return:  (B, 3, 3) or (3, 3)
    """
    if r.ndim == 1:  # Handle single vector case
        r = r.unsqueeze(0)  # Make it (1, 3) for uniform processing


    skew_r = vec2skew(r)  # (B, 3, 3)
    norm_r = r.norm(dim=-1, keepdim=True) + 1e-15  # (B, 1)
    eye = torch.eye(3, dtype=torch.float32, device=r.device).unsqueeze(0)  # (1, 3, 3)
    
    # Expand eye to match batch size
    eye = eye.expand(r.size(0), -1, -1)  # (B, 3, 3)

    # Compute rotation matrix
    sin_term = (torch.sin(norm_r) / norm_r).unsqueeze(-1)  # (B, 1, 1)
    cos_term = ((1 - torch.cos(norm_r)) / norm_r**2).unsqueeze(-1)  # (B, 1, 1)
    R = eye + sin_term * skew_r + cos_term * (skew_r @ skew_r)  # (B, 3, 3)


    #if R.size(0) == 1:  # If input was a single vector, return (3, 3)
    #    R = R.squeeze(0)
    return R


@dataclass
class KITTI360Dataset(BaseDataset):
    device: str = "cpu"
    split: str = "train"  # train, val, test, (refine)
    root_path: str = "data/kitti360"
    sequence_id: str = "4950"
    preload: bool = True  # preload data into GPU
    scale: float = 1      # scale to bounding box
    offset: list = field(default_factory=list)  # offset
    fp16: bool = True     # if preload, load into fp16.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    num_rays_lidar: int = 4096
    fov_lidar: list = field(default_factory=list)  # fov_up, fov [2.0, 26.9]
    z_offsets: list = field(default_factory=list)  # z_offset, z_offset_bot
    laser_offsets: list = field(default_factory=list)  # alpha_offset, alpha_offset_bot
    R: torch.Tensor = None
    T: torch.Tensor = None
    nmask: bool = False


    def __post_init__(self):
        if self.sequence_id == "1538":
            print("Using sequence 1538-1601")
            frame_start = 1538
            frame_end = 1601
        elif self.sequence_id == "1728":
            print("Using sequence 1728-1791")
            frame_start = 1728
            frame_end = 1791
        elif self.sequence_id == "1908":
            print("Using sequence 1908-1971")
            frame_start = 1908
            frame_end = 1971
        elif self.sequence_id == "3353":
            print("Using sequence 3353-3416")
            frame_start = 3353
            frame_end = 3416
        
        elif self.sequence_id == "2350":
            print("Using sequence 2350-2400")
            frame_start = 2350
            frame_end = 2400
        elif self.sequence_id == "4950":
            print("Using sequence 4950-5000")
            frame_start = 4950
            frame_end = 5000
        elif self.sequence_id == "8120":
            print("Using sequence 8120-8170")
            frame_start = 8120
            frame_end = 8170
        elif self.sequence_id == "10200":
            print("Using sequence 10200-10250")
            frame_start = 10200
            frame_end = 10250
        elif self.sequence_id == "10750":
            print("Using sequence 10750-10800")
            frame_start = 10750
            frame_end = 10800
        elif self.sequence_id == "11400":
            print("Using sequence 11400-11450")
            frame_start = 11400
            frame_end = 11450
        else:
            frame_start = int(self.sequence_id)
            frame_end = int(self.sequence_id) + 50
            #raise ValueError(f"Invalid sequence id: {sequence_id}")
        
        print(f"Using sequence {frame_start}-{frame_end}")
        self.frame_start = frame_start
        self.frame_end = frame_end


        self.training = self.split in ["train", "all", "trainval"]
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        if self.split == 'refine':
            self.split = 'train'
            self.num_rays_lidar = -1

      
        # load nerf-compatible format data.
        print("loading from", os.path.join(self.root_path, f"transforms_{self.sequence_id}_0000_all.json"))

        #check if it exists
        assert os.path.exists(os.path.join(self.root_path, f"transforms_{self.sequence_id}_0000_all.json")), "File not found"

        with open(
            os.path.join(self.root_path, 
                         f"transforms_{self.sequence_id}_0000_all.json"),
            "r",
        ) as f:
            transform = json.load(f)

        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"])
            self.W = int(transform["w"])
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])
            #create matrix for variable laser strenghts per laser ray with random init between 0 and 1 with gradient
            #self.laser_strength = torch.rand((self.H_lidar, 1), requires_grad=True)
            #init with 0

 


            #print gradient

        #self.z_offsets = [float(transform["z_offset"]), float(transform["z_offset_bot"])]

        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['lidar_file_path'])
        mask_filename = "train/advanced_mask.png"
        base_mask = cv2.imread(os.path.join(self.root_path, mask_filename), cv2.IMREAD_GRAYSCALE)
        base_mask = 1.0- torch.from_numpy(base_mask).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        lidar_base_mask = cv2.imread(os.path.join(self.root_path, "train/advanced_intensity_mask.png"), cv2.IMREAD_GRAYSCALE)
        lidar_base_mask = 1.0- torch.from_numpy(lidar_base_mask).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        if "val_ids" in transform:
            val_ids = transform["val_ids"]
            #get id of first frame
            frame0id = frames[0]["frame_id"]
            val_ids = [i-frame0id for i in val_ids]
            train_ids = [i for i in range(frame_start, frame_end+1)]
            #offset by frame0id
            train_ids = [i-frame0id for i in train_ids]
            train_ids = [i for i in train_ids if i not in val_ids]

            if self.split == 'train':
                self.selected_ids = train_ids

            else:
                self.selected_ids = val_ids

            print("Selected ids", self.selected_ids, self.split)


        else:
            print("No val_ids found")
            exit()


        self.poses_lidar = []
        self.images_lidar = []
        self.times = []
        self.base_mask = base_mask
        self.lidar_base_mask = lidar_base_mask

        for f in tqdm.tqdm(frames, desc=f"Loading {self.split} data"):
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]


            f_lidar_path = os.path.join(self.root_path, f["lidar_file_path"])
            f_lidar_path=f_lidar_path.replace("_0000", "")
            # channel1 incidence, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path)
            mask_thresh=0.0
            if self.nmask:
                mask_thresh = 0.01

            ray_drop = np.where(pc.reshape(-1, 3)[:, 0] <= mask_thresh, 0.0, 1.0).reshape(
                self.H_lidar, self.W_lidar, 1
            )

            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale, pc[:, :, 0, None]],
                axis=-1,
            )

            time = np.asarray((f['frame_id']-frame_start)/(frame_end-frame_start))
            
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
            self.times.append(time)

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (
            self.poses_lidar[:, :3, -1] - self.offset
        ) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        self.images_lidar = torch.from_numpy(np.stack(self.images_lidar, axis=0)).float()  # [N, H, W, C]

        self.times = torch.from_numpy(np.asarray(self.times, dtype=np.float32)).view(-1, 1) # [N, 1]

        if self.preload:
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.fp16:
                dtype = torch.half
            else:
                dtype = torch.float
            self.images_lidar = self.images_lidar.to(dtype).to(self.device)
            self.times = self.times.to(self.device)

        self.intrinsics_lidar = self.fov_lidar


    def collate(self, index):
        B = len(index)  # a list of length 1

        results = {}


        poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]


        R = self.R[index]
        T = self.T[index]
        
        R = Exp(R)  # (B, 3, 3)
        pose_off = torch.cat([R, T.unsqueeze(-1)], dim=-1)
        #make 3x4 matrix
        pose_off = torch.cat([pose_off, torch.zeros_like(pose_off[:, 0:1])], dim=1)  # (B, 4, 4)
        pose_off[:, 3, 3] = 1.0
        poses_lidar = torch.matmul(poses_lidar, pose_off)
  


        prev_index = [i-1 for i in index]
        next_index = [i+1 for i in index]

        #clip to valid range
        prev_index = [max(0, i) for i in prev_index]
        next_index = [min(len(self.poses_lidar)-1, i) for i in next_index]

       # print(prev_index, next_index, index)
  

        prev_poses_lidar = self.poses_lidar[prev_index].to(self.device)  # [B, 4, 4]
        next_poses_lidar = self.poses_lidar[next_index].to(self.device)  # [B, 4, 4]

       
        prev_R = self.R[prev_index]
        prev_T = self.T[prev_index]
        
        prev_R = Exp(prev_R)  # (B, 3, 3)
        
        prev_pose_off = torch.cat([prev_R, prev_T.unsqueeze(-1)], dim=-1)
        prev_pose_off = torch.cat([prev_pose_off, torch.zeros_like(prev_pose_off[:, 0:1])], dim=1)  # (B, 4, 4)
        prev_pose_off[:, 3, 3] = 1.0
        prev_poses_lidar = torch.matmul(prev_poses_lidar, prev_pose_off)


        next_R = self.R[next_index]
        next_T = self.T[next_index]

        next_R = Exp(next_R)  # (B, 3, 3)

        next_pose_off = torch.cat([next_R, next_T.unsqueeze(-1)], dim=-1)
        next_pose_off = torch.cat([next_pose_off, torch.zeros_like(next_pose_off[:, 0:1])], dim=1)  # (B, 4, 4)
        next_pose_off[:, 3, 3] = 1.0
        next_poses_lidar = torch.matmul(next_poses_lidar, next_pose_off)
 

        rays_lidar = get_lidar_rays(
            poses_lidar,
            self.intrinsics_lidar,
            self.H_lidar,
            self.W_lidar,
            self.z_offsets,
            self.laser_offsets,
            self.num_rays_lidar,
            self.patch_size_lidar,
            self.scale
        )


        time_lidar = self.times[index].to(self.device) # [B, 1]

        images_lidar = self.images_lidar[index].to(self.device)  # [B, H, W, 3]

        if self.training:
            C = images_lidar.shape[-1]
            images_lidar = torch.gather(
                images_lidar.view(B, -1, C),
                1,
                torch.stack(C * [rays_lidar["inds"]], -1),
            )  # [B, N, 3]

        results.update(
            {
                "H_lidar": self.H_lidar,
                "W_lidar": self.W_lidar,
                "rays_o_lidar": rays_lidar["rays_o"],
                "rays_d_lidar": rays_lidar["rays_d"],
                "images_lidar": images_lidar,
                "time": time_lidar,
                "poses_lidar": poses_lidar,
                "poses_before": prev_poses_lidar,
                "poses_after": next_poses_lidar,
                "row_inds": rays_lidar["row_inds"],
                "col_inds": rays_lidar["col_inds"],
                "index": index,
                "base_mask": self.base_mask,
                "lidar_base_mask": self.lidar_base_mask,
            }
        )

        return results

    def dataloader(self):
        size = len(self.poses_lidar)#-2
        print("size", size)
        #if self.val_ids exist
        if hasattr(self, 'selected_ids'):
            indices = self.selected_ids
            #to list
            indices = list(indices)
            #print(indices, list(range(size)))
        else:
            indices = list(range(size))
            print(indices)
            print("Error: No val_ids found in transform file. Please check the file.")
            exit()

        #list(range(1,size+1)),
        #list(range(size)),
        loader = DataLoader(
            indices,
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)#-2
        return num_frames
