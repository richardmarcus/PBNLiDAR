
This repository is the official PyTorch implementation for the paper Physically Based Neural LiDAR Resimulation accepted at [IEE ITSC 2025](https://ieee-itsc.org/2025/).



## Abstract
Methods for Novel View Synthesis (NVS) have recently found traction in the field of LiDAR simulation and large-scale 3D scene reconstruction. While solutions for faster rendering or handling dynamic scenes have been proposed, LiDAR specific effects remain insufficiently addressed. By explicitly modeling sensor characteristics such as rolling shutter, laser power variations, and intensity falloff, our method achieves more accurate LiDAR simulation compared to existing techniques. We demonstrate the effectiveness of our approach through quantitative and qualitative comparisons with state-of-the-art methods, as well as ablation studies that highlight the importance of each sensor model component. Beyond that, we show that our approach exhibits advanced resimulation capabilities, such as generating high resolution LiDAR scans in the camera perspective.


## Table of Contents
We first provide general interesting conclusions from our paper, including an upscaled LiDAR dataset and then give instructions how to set up our system.
- [Improved LiDAR Intrinsics](#improved-lidar-intrinsics)
- [Improved Masking](#improved-masking)
- [HD LiDAR Reconstruction](#hd-lidar-reconstruction)
- [Getting started](#getting-started)
    - [🛠️ Installation](#️-installation)
    - [📁 Dataset](#-dataset)
    - [Training](#training)
    - [Simulation](#simulation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)


## Improved LiDAR Intrinsics
One contribution of our paper is the reconstruction of the correct LiDAR intrinsics. This can also be helpful for other tasks that use range-view panoramic images.


<img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/laser_offsets3.png" width=65%>
<p align="left"><i>Visualization of laser pattern for Velodyne HDL-64E: two optical centers with different FOVs.</i></p>

We have added our reconstructed parameters in intrinsics.json and a standalone script to showcase their use to perform the projection task:
```bash
python debug_dual_projection.py
```

With these, there are no longer gaps in the projection, see raydrop masks below.

<img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/gt_mask_old.png" width=65%>
<p align="left"><i>Raydrop mask used in previous work</i></p>

<img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/gt_mask.png" width=65%>
<p align="left"><i>Improved raydrop mask</i></p>



## Improved Masking
This further allows our system to only predict actual ray misses by constructing a statistical global raydrop mask that only includes ego vehicle occlusion:

<img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/advanced_mask.png" width=65%>

For intensity, we notice invalid measurements, for which we create a special mask.

<img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/advanced_intensity_mask.png" width=65%>

## HD LiDAR Reconstruction
For more details regarding the Physically Based LiDAR pipeline, please refer to the paper.
One interesting use case, however, is the possibility to generate a high resolution reconstruction of the LiDAR return in the camera perspective.
Thereby, we also separate the raw intensity output into a base "albedo" intensity and a reflectivity map.
This could be used for image2image translation between the camera and the LiDAR.

[The dataset is available via Zenodo.](https://zenodo.org/records/16685170) and consists of more than 10k frames.
Beyond the mentioned modalities and the combined intensity, we include our computed incidence maps, raydrop output from the NeRF, and (instance) segmentation maps from KITTI-360.

<table>
<tr>
<td><img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/lidar4d_0304_0000.png" width="100%"><p align="center"><i>Reference Camera Image</i></p></td>
<td><img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/lidar4d_0304_0003.png" width="100%"><p align="center"><i>Depth Map</i></p></td>
</tr>
<tr>
<td><img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/lidar4d_0304_0002.png" width="100%"><p align="center"><i>Albedo Intensity</i></p></td>
<td><img src="https://github.com/richardmarcus/PBNLiDAR/releases/download/v1.0.0/lidar4d_0304_0001_grey.png" width="100%"><p align="center"><i>Reflectivity Map</i></p></td>
</tr>
</table>


## Getting started


### 🛠️ Installation

```bash
git clone https://github.com/richardmarcus/PBNLiDAR.git
cd PBNLiDAR

conda create -n pbl python=3.9
conda activate pbl

# PyTorch
# CUDA 12.1
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
 pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Dependencies
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# navigate back to root dir
cd -

# compile packages in utils
cd utils/chamfer3D
python setup.py install
cd -

```


### 📁 Dataset
#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/c9f5d5c5-ac48-4d54-8109-9a8b745bbca0" width=65%>  

Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`.  
(or use symlinks: `ln -s DATA_ROOT/KITTI-360 ./data/kitti360/`).  
The folder tree is as follows:  

```bash
data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
```

Next, run KITTI-360 dataset preprocessing: (set `DATASET` and `SEQ_ID`)  

```bash
bash preprocess_data.sh
```
After preprocessing, your folder structure should look like this:  

```bash
configs
├── kitti360_{sequence_id}.txt
data
└── kitti360
    ├── KITTI-360
    │   ├── calibration
    │   ├── data_3d_raw
    │   └── data_poses
    ├── train
    ├── transforms_{sequence_id}test.json
    ├── transforms_{sequence_id}train.json
    └── transforms_{sequence_id}val.json
```

### Training

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
The following script uses an example configruation for run_kitti_pbl.sh
```bash
# KITTI-360
bash train_scene.sh
```

### Simulation
After reconstruction, you can use the simulator to render and manipulate LiDAR point clouds in the whole scenario. It supports dynamic scene re-play, novel LiDAR configurations (`--fov_lidar`, `--H_lidar`, `--W_lidar`) and novel trajectory (`--shift_x`, `--shift_y`, `--shift_z`).  

We provide simulation scripts for the base config and HD camera rays to get high resolution output(run_kitti_pbl_sim_lidar.sh and run_kitti_pbl_sim_cam.sh).

Check the sequence config and corresponding workspace and model path (`--ckpt`).  
Run the following command:
```bash
bash run_kitti_pbl_sim_cam.sh
```
The results will be saved in the workspace folder.


## Citation ([arXive Postprint](https://arxiv.org/abs/2507.12489))

```

@misc{marcus2025physicallybasedneurallidar,
      title={Physically Based Neural LiDAR Resimulation}, 
      author={Richard Marcus and Marc Stamminger},
      year={2025},
      eprint={2507.12489},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.12489}, 
}

```


## Acknowledgement
We sincerely appreciate the great contribution of [LiDAR4D](https://github.com/ispc-lab/LiDAR4D), which our implementation and this Readme is based on.


## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
