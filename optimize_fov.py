from matplotlib import pyplot as plt
import torch
import numpy as np
import cv2
import os
from utils.convert import  calculate_ring_ids, compare_lidar_to_pano_with_intensities, compare_lidar_to_pano_with_intensities_torch, lidar_to_pano_with_intensities
device = "cuda" if torch.cuda.is_available() else "cpu"

path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

path_mc = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_deskewed/"

fov_x = 360
fov_y = 32
ray_res_x =1024
ray_res_y = 64
max_files = 1
opt_crop = 3
opt_steps =  0000
int_off = 0
opt_indivual = True

files = os.listdir(path)
files.sort()

#initial offsets (to be optimized)
K = [2.02984126984, 11.0317460317, -8.799812, 16.541]
z_offsets = [-0.202, -0.121]

#z_offsets = [ 0,0]
#K = [2, 13, -11, 13]
K = [ 1.9535843, 11.027419,  -9.015155,  16.532858 ]
z_offsets = [-0.20289962, -0.1270004 ]


'''laser_offsets [ 0.0101472   0.02935141 -0.04524597  0.04477938 -0.00623795  0.04855699
 -0.02581356 -0.00632023  0.00133613  0.05607248  0.00494516  0.00062785
  0.03141189  0.02682017  0.01036519  0.02891498 -0.01124913  0.04208804
 -0.0218643   0.00743873 -0.01018788 -0.01669445  0.00017374  0.0048293
  0.03166919  0.03558188  0.01552001 -0.03950449  0.00887087  0.04522041
 -0.04557779  0.01275884  0.02858396  0.06113308  0.03508026 -0.07183428
 -0.10038704  0.02749107  0.0291795  -0.03833354 -0.07382096 -0.14437623
 -0.09460489 -0.0584761   0.01881664 -0.02696179 -0.02052307 -0.15732896
 -0.03719316 -0.00687183  0.07373429  0.03398049  0.04429062 -0.05352834
 -0.07988049 -0.02726229 -0.00934669  0.09552395  0.0850026  -0.00946006
 -0.05684165  0.0798225   0.10324192  0.08222152]'''


laser_offsets = [0.0101472, 0.02935141, -0.04524597,  0.04477938, -0.00623795,  0.04855699, 
    -0.02581356, -0.00632023,  0.00133613,  0.05607248,  0.00494516,  0.00062785,
    0.03141189,  0.02682017,  0.01036519,  0.02891498, -0.01124913,  0.04208804,
    -0.0218643,   0.00743873, -0.01018788, -0.01669445,  0.00017374,  0.0048293,
    0.03166919,  0.03558188,  0.01552001, -0.03950449,  0.00887087,  0.04522041,
    -0.04557779,  0.01275884,  0.02858396,  0.06113308,  0.03508026, -0.07183428,
    -0.10038704,  0.02749107,  0.0291795,  -0.03833354, -0.07382096, -0.14437623,
    -0.09460489, -0.0584761,   0.01881664, -0.02696179, -0.02052307, -0.15732896,
    -0.03719316, -0.00687183,  0.07373429,  0.03398049,  0.04429062, -0.05352834,
    -0.07988049, -0.02726229, -0.00934669,  0.09552395,  0.0850026,  -0.00946006,
    -0.05684165,  0.0798225,   0.10324192, 0.08222152]

#laser_offsets = np.zeros(64)
#Optimized K = [ 1.9647572 11.0334425 -8.979475  16.52717  ]
#Optimized z_offsets = [-0.20287499 -0.12243641]

K = [ 1.9647572, 11.0334425, -8.979475,  16.52717 ]
z_offsets = [-0.20287499, -0.12243641 ]

all_azi = []
all_pcds = []
all_pcds2 = []
num_files = 0
all_yrings = []
all_yrings2 = []
all_intens = []
all_intens2 = []
all_cids = []
all_cids2 = []
all_azi2 = []
all_masks = []
all_masks2 = []
all_weights = []
all_weights2 = []

cid = 0
for file in files:
    if int(file.split(".")[0]) < (1552):
        continue
    print(file)
    bin_pcd = np.fromfile(os.path.join(path, file), dtype=np.float32).reshape(-1,4)
    #bin_pcd = bin_pcd[:, :3]
   # bin_mc = np.fromfile(os.path.join(path_mc, file), dtype=np.float32).reshape(-1,3)
    #print mean distance between points from both point clouds
  #  print(np.linalg.norm(bin_pcd[:, :3] - bin_mc, axis=1).max())

    #overwrite with deskewed
 #   bin_pcd[:, :3] = bin_mc
    y_r = calculate_ring_ids(bin_pcd, ray_res_y)

    if y_r.min() != 0:
        ys, ys2= compare_lidar_to_pano_with_intensities(bin_pcd, ray_res_y, ray_res_x, K, z_offsets, ring=False)
        ys = np.concatenate((ys, ys2+32))
        avg_ys = np.mean(ys)
        avg_yr = np.mean(y_r)
        print("Diff", avg_ys - avg_yr)
        rounded_diff = np.round(avg_ys - avg_yr)
        y_r = y_r + rounded_diff

    intensities = bin_pcd[:, 3]

    bin_pcd = bin_pcd[:, :3]
    distances = np.linalg.norm(bin_pcd, axis=1)
    azimuths = np.arctan2(bin_pcd[:, 1], bin_pcd[:, 0])
    mask = np.logical_and(azimuths >= -np.pi/opt_crop, azimuths <= np.pi/opt_crop)
    mask = np.logical_and(mask, distances <=80)

    #only keep if yr is between 25 and 35
    #mask2 = np.logical_and(y_r >= 25, y_r <= 35)
    #mask = np.logical_and(mask, mask2)

    azimuths = azimuths[mask]
    intensities = intensities[mask]
    weights = 1-np.abs((azimuths/np.pi))
    #weights = weights*weights



    bin_pcd = bin_pcd[mask]
    y_r = y_r[mask]
    cids = np.ones_like(y_r)*cid
   

    mask = y_r < ray_res_y//2
    ys_ring = y_r[mask]
    ys_ring2 = y_r[~mask]

    weights2 = weights[~mask]
    weights = weights[mask]

    all_weights.append(weights)
    all_weights2.append(weights2)


    azi2 = azimuths[~mask]
    azimuths = azimuths[mask]
    intensities2 = intensities[~mask]
    intensities = intensities[mask]
    cids2 = cids[~mask]
    cids = cids[mask]
    all_cids.append(cids)
    all_intens.append(intensities)

    all_cids2.append(cids2)
    all_intens2.append(intensities2)

    all_azi2.append(azi2)


    all_masks.append(mask[mask])
    all_masks2.append(mask[~mask])

    all_pcds.append(bin_pcd[mask])
    all_pcds2.append(bin_pcd[~mask])


    all_yrings.append(ys_ring)
    all_yrings2.append(ys_ring2)
    all_azi.append(azimuths)


    cid += 1

    num_files += 1
    if num_files == max_files:
        break


ys_ring = np.concatenate(all_yrings)
ys_ring2 = np.concatenate(all_yrings2)
ys_ring = np.concatenate((ys_ring, ys_ring2))

bin_pcd = np.concatenate(all_pcds)
bin_pcd2 = np.concatenate(all_pcds2)
bin_pcd = np.concatenate((bin_pcd, bin_pcd2))
mask = np.concatenate(all_masks)
mask2 = np.concatenate(all_masks2)
mask = np.concatenate((mask, mask2))

azimuths = np.concatenate(all_azi)
intensities = np.concatenate(all_intens)
cids = np.concatenate(all_cids)

azimuths2 = np.concatenate(all_azi2)
intensities2 = np.concatenate(all_intens2)
cids2 = np.concatenate(all_cids2)

azimuths = np.concatenate((azimuths, azimuths2))
intensities = np.concatenate((intensities, intensities2))
cids = np.concatenate((cids, cids2))

weights = np.concatenate(all_weights)
weights2 = np.concatenate(all_weights2)
weights = np.concatenate((weights, weights2))

#move mask to torch
mask = torch.tensor(mask, dtype=torch.bool, device=device)

weights = torch.tensor(weights, dtype=torch.float32, device=device)


param_K = torch.tensor(K, dtype=torch.float32, device=device, requires_grad=True)
param_z = torch.tensor(z_offsets, dtype=torch.float32, device=device, requires_grad=True)
param_laser_offsets_inner= torch.tensor(laser_offsets, dtype=torch.float32, device=device, requires_grad=opt_indivual)

#add 0 to beginning of laser_offsets and to end, make sure that the gradient is not lost
#param_laser_offsets = torch.cat((torch.tensor([0], dtype=torch.float32, device=device), param_laser_offsets_inner, torch.tensor([0], dtype=torch.float32, device=device)))

param_laser_offsets = param_laser_offsets_inner

ys_ring_t = torch.tensor(ys_ring, dtype=torch.float32, device=device)
bin_pcd = torch.tensor(bin_pcd, dtype=torch.float32, device=device)

def compute_loss(ys_cat, ys_ring_t):

    d_y = torch.abs(ys_ring_t - ys_cat +int_off)
    #d_y*= 2*weights*weights
    return d_y.mean(), ys_cat

def compute_loss3(ys_cat, ys_ring_t):

    mse = torch.nn.MSELoss()
    #rmse = torch.sqrt(mse(ys_cat, ys_ring_t))
    loss = mse(ys_cat, ys_ring_t)
    return loss, ys_cat

def compute_loss2(ys_cat, ys_ring_t):

    #bad
    d_y = torch.abs(ys_ring_t - ys_cat)
    ring_ids = ys_ring_t.long()
    sum_dy = torch.zeros(64, device=device)
    count_dy = torch.zeros(64, device=device)

    sum_dy.index_add_(0, ring_ids, d_y)
    count_dy.index_add_(0, ring_ids, torch.ones_like(d_y))
    ring_mean = sum_dy / (count_dy + 1e-8)
    #select 90% of closest points
    distance = torch.abs(d_y - ring_mean[ring_ids])
    maskr = distance < distance.kthvalue(int(0.1*len(distance)))[0]
    d_y_without_outliers = d_y[maskr]

    return d_y_without_outliers.mean(), ys_cat

def compute_loss4(ys_cat, ys_ring_t):

    d_y = torch.abs(ys_ring_t - ys_cat)
    ring_ids = ys_ring_t.long()
    sum_dy = torch.zeros(64, device=device)
    count_dy = torch.zeros(64, device=device)

    sum_dy.index_add_(0, ring_ids, d_y)
    count_dy.index_add_(0, ring_ids, torch.ones_like(d_y))
    ring_mean = sum_dy / (count_dy + 1e-8)
    #select 90% of closest points
    distance = torch.abs(d_y - ring_mean[ring_ids])
    maskr = distance < distance.kthvalue(int(0.5*len(distance)))[0]
    mse_without_outliers = torch.nn.MSELoss()
    loss = mse_without_outliers(ys_cat[maskr], ys_ring_t[maskr])

    return loss, ys_cat


cur_loss = 0
loss_checker = 0


optimizer = torch.optim.Adam([param_K, param_z, param_laser_offsets_inner], lr=0.02)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100, verbose=True)


print(opt_steps)
for i in range(opt_steps):
    optimizer.zero_grad()
    spread_laser_offsets = param_laser_offsets[ys_ring_t.long()]
    ys, ys2 = compare_lidar_to_pano_with_intensities_torch(bin_pcd, ray_res_y, ray_res_x, param_K, param_z, ring=False, mask=mask, laser_offsets=spread_laser_offsets)
    ys_cat = torch.cat((ys, ys2 + 32), dim=0)
    loss, opt_ys = compute_loss(ys_cat=ys_cat, ys_ring_t=ys_ring_t)

    if False:
        d_y =  ys_cat
        ring_ids = ys_ring_t.long()
        sum_dy = torch.zeros(64, device=device)
        count_dy = torch.zeros(64, device=device)

        sum_dy.index_add_(0, ring_ids, d_y)
        count_dy.index_add_(0, ring_ids, torch.ones_like(d_y))
        ring_mean = sum_dy / (count_dy + 1e-8)

        inter_ring_delta = torch.abs(ring_mean[:63] - ring_mean[1:64])
        inter_ring_loss = torch.abs(torch.ones_like(inter_ring_delta) - inter_ring_delta).mean()
        loss += 0.1*inter_ring_loss 
        print("Inter ring loss", inter_ring_loss)
  
    #loss_all = torch.abs(ys_ring_t.mean() - opt_ys.mean())
    #loss += loss_all *0.1
    #z_diff = ((param_z[1] - param_z[0]) - 0.08).abs()
    #loss += z_diff

    #param_sum = param_laser_offsets.mean()
    #loss += param_sum.abs() * 0.1
    #param_abs = param_laser_offsets.abs().mean()
    #loss += param_abs * .1

   # end_loss = torch.abs(param_laser_offsets[-1])
    #loss += end_loss * 0.1

    #z_diff = param_z[1] - param_z[0]
    # Force z_diff to be greater than 0
    #loss += torch.relu(-z_diff)


    loss.backward()
    optimizer.step()
    scheduler.step(loss)



    delta_loss = np.abs(cur_loss - loss.detach().cpu().numpy())
    ys = opt_ys.detach().cpu().numpy()
   # print(param_laser_offsets.mean())

    #move to cpu 
    cur_loss = loss.detach().cpu().numpy()


    if delta_loss < 0.00001:
        #print(delta_loss, loss_checker)
        loss_checker += 1
    else:
        loss_checker = 0

    if loss_checker == 10:
        print("Converged")
        print("Loss =", cur_loss, i)
            #print learning rate
        for param_group in optimizer.param_groups:
            print("lr2", param_group['lr'])
        break
       

print("Optimized K =", param_K.detach().cpu().numpy())
print("Optimized z_offsets =", param_z.detach().cpu().numpy())
print("Delta Z", param_z.detach().cpu().numpy()[1]-param_z.detach().cpu().numpy()[0])
print("Optimized laser_offsets", param_laser_offsets.detach().cpu().numpy())

bin_pcd = bin_pcd.detach().cpu().numpy()
mask = mask.detach().cpu().numpy()
weights = weights.detach().cpu().numpy()

K = param_K.detach().cpu().numpy()
z_offsets = param_z.detach().cpu().numpy()
spread_laser_offsets = param_laser_offsets[ys_ring_t.long()]


ys, ys2 = compare_lidar_to_pano_with_intensities(bin_pcd, ray_res_y, ray_res_x, K, z_offsets, ring=False, mask=mask, laser_offsets=spread_laser_offsets.detach().cpu().numpy(), rids = ys_ring)
ys = np.concatenate((ys, ys2+32))-int_off

#print(list(zip(ys.tolist(), spread_laser_offsets.tolist())))


mean_diff = np.abs(ys_ring - ys).mean()
print("Mean diff", mean_diff)


mean_diff2 = np.mean(ys_ring) - np.mean(ys)
print("Mean diff2", mean_diff2)

mean_diff3 = np.mean(param_laser_offsets.detach().cpu().numpy())
print("offset mean", mean_diff3)

print(cids.shape, intensities.shape, ys.shape)
print(np.min(cids), np.max(cids))
plt.figure( dpi=1000)
#make figure really wide
plt.figure(figsize=(20, 5))
for i in range(ray_res_y):
    
    mask = ys_ring == i
    
    ys_ring_masked = ys_ring[mask]
    ys_masked = ys[mask]
    x_values = np.arange(0, len(ys_masked))
    #to float
    x_values = x_values.astype(np.float32)
    x_values = -azimuths[mask] + 2*np.pi/opt_crop*cids[mask] 
  
    colors = intensities[mask]

    #colors = cids[mask]


    #set dpi to 1000
    plt.scatter(x_values, (63-ys_masked), c= colors, s=0.1)
    plt.plot(x_values, 63-ys_ring_masked, color="red", linewidth=0.2)
    #lt.scatter(x_values, 63-ys_ring_masked, c= colors, s=0.1, cmap="plasma")
    #plt.scatter(x_values,weights[mask]+ys_ring_masked, color="green", s=0.5)
    


plt.savefig(f"lidar_pano.png")
print("Saved lidar_pano.png")

#clear figure and make new
plt.clf()

plt.figure(figsize=(5, 2))
#plot laser offsets and draw dots and line
plt.plot(param_laser_offsets.detach().cpu().numpy(), marker="o")

plt.savefig("plots/laser_offsets.png")
print("Saved plots/laser_offsets.png")


#concatenate intensities to bin_pcd
bin_pcd = np.concatenate((bin_pcd, intensities[:, None]), axis=1)

if max_files == 1:
    pano, intensities = lidar_to_pano_with_intensities(bin_pcd, ray_res_y, ray_res_x, K, z_offsets, ring=False)

    cv2.imwrite("pano.png", intensities*255)

