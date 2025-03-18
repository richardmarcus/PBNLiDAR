
import os
from matplotlib import pyplot as plt
import numpy as np

path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"
files = os.listdir(path)
files.sort()

for file in files:
    if int(file.split(".")[0]) < (1551):
        continue
    print(file)
    bin_pcd = np.fromfile(os.path.join(path, file), dtype=np.float32).reshape(-1,4)
    
    #sample 100 random points
    #seed
    np.random.seed(0)
    bin_pcd = bin_pcd[np.random.choice(bin_pcd.shape[0], 10, replace=False), :]
    azimuths = np.arctan2(bin_pcd[:, 1], bin_pcd[:, 0])
    plt.figure( dpi=100)

    plt.figure(figsize=(5, 5))
    x_values = azimuths
    plt.scatter(x_values,np.abs(x_values), color="green")
    #save
    #print x values and y values
    print(x_values)
    print(np.abs(x_values))

    plt.savefig("test_plot.png")
    break