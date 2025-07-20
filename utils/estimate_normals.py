import cv2
import numpy as np
import open3d as o3d
def build_normal_xyz(xyz,smooth_normals=True, use_o3d=True, replace_o3d=False):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
     '''
    

  
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

 

    dists = np.linalg.norm(xyz, axis=2)
    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0)    
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0)    
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0)    
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1)

    # build cross product
    normal = -np.dstack((Syx*Szy - Szx*Syy,
                        Szx*Sxy - Szy*Sxx,
                        Sxx*Syy - Syx*Sxy))

    # normalize cross product
    n = np.linalg.norm(normal, axis=2)
    #set 0 to 1 to avoid division by zero
    n[n ==0] = 1
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    #for image border copy the normal from the nearest pixel
    normal[0, :, :] = normal[1, :, :]
    normal[-1, :, :] = normal[-2, :, :]
    normal[:, 0, :] = normal[:, 1, :]
    normal[:, -1, :] = normal[:, -2, :]

    

    #select all normals whose neighbors above and below have a distance of less than 0.1
 
    # Apply bilateral filter to smooth normals while preserving edges

    if smooth_normals: 

        gradient_y = np.abs(Sxy) + np.abs(Syy) + np.abs(Szy)
        # Define a threshold to distinguish weak edges
        weak_edge_threshold = 8 # Adjust this value as needed

        # Create a mask for weak edges
        weak_edges_mask = (gradient_y < weak_edge_threshold)

        distance_mask = dists < 40

        strong_edge_mask = gradient_y > 20

        upward_mask = normal[:, :, 2] > 0.95
        weak_upward_mask = normal[:, :, 2] > .98
    
        #weak_upward_mask = np.logical_and(upward_mask, gradient_y > weak_edge_threshold*.5)

        upward_mask = np.logical_and(upward_mask,strong_edge_mask)
        upward_mask = np.logical_and(upward_mask, distance_mask)

        weak_edges_mask = np.logical_and(weak_edges_mask, distance_mask)
        weak_upward_mask = np.logical_and(weak_upward_mask, distance_mask)

        # Apply Gaussian blur to normals in weak edge regions
        blurred_normal = cv2.GaussianBlur(normal.astype(np.float32), (1, 3), 0)  # Adjust kernel size and sigma as needed

        # Use the mask to blend the original normals and blurred normals
        #normal[weak_edges_mask] = blurred_normal[weak_edges_mask]

        normal[upward_mask] = (0,0,1)#+ 0.1*blurred_normal[upward_mask]

        weak_upward_mask = np.logical_and(weak_upward_mask, np.logical_not(upward_mask))
        
        normal[weak_upward_mask] =np.mean(normal[weak_upward_mask],axis=0)*0.5+0.5*blurred_normal[weak_upward_mask]

        weak_edges_mask = np.logical_and(weak_edges_mask, np.logical_not(upward_mask))
        weak_edges_mask = np.logical_and(weak_edges_mask, np.logical_not(weak_upward_mask))


        normal[weak_edges_mask] = blurred_normal[weak_edges_mask]

        #normalize normals
        n = np.linalg.norm(normal, axis=2)
        #set 0 to 1 to avoid division by zero
        n[n ==0] = 1
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n



    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(Sxx**2 + Sxy**2 + Syx**2 + Syy**2 + Szx**2 + Szy**2)
    # Threshold for strong gradients (adjust as needed)
    strong_gradient_threshold = 30
    #convert to point cloud
    if use_o3d:

        pcd = o3d.geometry.PointCloud()
        pcd_points = np.reshape(xyz, (-1, 3))
        pcd_normals =  np.reshape(-xyz, (-1, 3))
        #pcd_normals = np.reshape(normal, (-1, 3))

        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
        #estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        d_normal = np.asarray(pcd.normals)
        d_normal = np.reshape(d_normal, (xyz.shape[0], xyz.shape[1], 3))

        # Selectively replace normals based on gradient magnitude
        mask = gradient_magnitude > strong_gradient_threshold
        mask = np.logical_and(mask, dists<20)
        normal[mask] = d_normal[mask]

    if replace_o3d:
        pcd = o3d.geometry.PointCloud()
        pcd_points = np.reshape(xyz, (-1, 3))
        pcd_normals =  np.reshape(-xyz, (-1, 3))
        #pcd_normals = np.reshape(normal, (-1, 3))

        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
        #estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=200))
        d_normal = np.asarray(pcd.normals)
        d_normal = np.reshape(d_normal, (xyz.shape[0], xyz.shape[1], 3))
        normal = d_normal

    
    return normal
