import os
log_path="/home/oq55olys/Cluster/LiDAR4D/log/"

#list first folder
folders = os.listdir(log_path)
for folder in folders:
    print (folder)
    #check if folder name starts with "LiDAR4D"
    if not folder.startswith("kitti360_lidar4d"):
        continue
    #check if folder name contains "_0000"
    if "_0000" in folder:
        continue
    #check if folder name contains "_"
    if "_" not in folder:
        continue

    #get third element divided by _ and add _0000
    sequnce = folder.split("_")[2]


    #rename folder
    #os.rename(log_path + folder, log_path + folder.replace(sequnce, sequnce + "_0000"))

    print("####")
