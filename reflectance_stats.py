import torch
import numpy as np
import cv2
import os

from matplotlib import pyplot as plt



device = "cuda" if torch.cuda.is_available() else "cpu"

path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/train_0000"


files = os.listdir(path)
#only take those with .npy extension
files = [file for file in files if file.endswith(".npy")]
files.sort()
#max_files = 
#take max files randomly not the first 10
#random_files = np.random.choice(files, max_files, replace=False)
#files = random_files

def plot_intensity_for_distance_for_incidences(bucket_size=0.1, bin_size=4):

    # Calculate the number of buckets needed to cover the range of incidences
    num_buckets = int(1 / bucket_size) + 1  # Cover incidences up to 1

    # create empty list with the calculated number of elements
    intensity_buckets = [[] for _ in range(num_buckets)]
    distance_buckets = [[] for _ in range(num_buckets)]

    for i, file in enumerate(files):

        pano = np.load(os.path.join(path, file))
        pano = pano.astype(np.float32)
        #get the intensities
        incidences = pano[:, :, 0]
        distances = pano[:, :, 2]
        intensities = pano[:, :, 1]
        #distance_buckets cast to int
        incidence_buckets = np.round(incidences / bucket_size).astype(int)
        

        # Flatten the arrays and filter based on conditions
        valid_indices = (distances.flatten() <= 80) & (intensities.flatten() <= 2) & (incidences.flatten() > 0) & (incidences.flatten() < .99)
        
        
        valid_distances = distances.flatten()[valid_indices]
        valid_incidences = incidence_buckets.flatten()[valid_indices]
        valid_intensities = intensities.flatten()[valid_indices] #/(valid_incidences*0.05+1)

        
        # Append the valid intensities and distances to their respective buckets
        for incidence in range(num_buckets):
            indices = (valid_incidences == incidence)
            intensity_buckets[incidence].extend(valid_intensities[indices])
            distance_buckets[incidence].extend(valid_distances[indices])



    k = 0

    for bucket, distance_bucket in zip(intensity_buckets, distance_buckets):
        # Calculate mean intensity for each bin_size distance bucket
        bucket_intensities = np.array(bucket)
        bucket_distances = np.array(distance_bucket)

        if len(bucket_intensities) == 0:
            continue
        # Create distance buckets of size bin_size
        distance_bins = np.arange(0, np.max(bucket_distances) + bin_size, bin_size)
        
        # Calculate mean intensity for each distance bin
        mean_intensities = []
        distance_bin_centers = []
        # Vectorize the binning and mean calculation
        bin_indices = np.digitize(bucket_distances, distance_bins) - 1  # Subtract 1 to align with array indices
        
        # Filter out-of-bounds indices
        valid_bins = (bin_indices >= 0) & (bin_indices < len(distance_bins) - 1)
        bin_indices = bin_indices[valid_bins]
        intensities_in_bins = bucket_intensities[valid_bins]

        
        # Calculate the mean intensity for each bin using bincount and groupby
        bin_counts = np.bincount(bin_indices, minlength=len(distance_bins) - 1)
        bin_sums = np.bincount(bin_indices, weights=intensities_in_bins, minlength=len(distance_bins) - 1)
        
        # Avoid division by zero
        #set binsumss to 1 if bin counts is 0
        #bin_counts = np.where(bin_counts == 0, 1, bin_counts)
        mean_intensities = np.where(bin_counts > 0, bin_sums / bin_counts, 0)

        #max_intensities = np.where(bin_counts > 0, np.max(intensities_in_bins), 0)
        #mean_intensities = max_intensities
        # Calculate bin centers
        distance_bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        # Filter out bins with no data
        valid_bins = bin_counts > 0
        mean_intensities = mean_intensities[valid_bins]
        distance_bin_centers = distance_bin_centers[valid_bins]

        #compute greyscale based on k and num_buckets
        greyscale = k/num_buckets
        print(k, num_buckets)

        # Plot mean intensities vs distance bin centers
        #use viridis colormap
        plt.plot(distance_bin_centers, mean_intensities, 'o-', markersize=5, color=plt.cm.viridis(greyscale))
        k+=1


    plt.xlabel("Distance (m)")
    plt.ylabel("Mean Intensity")
    plt.ylim([0, 0.5])

    #plot colorbar
    ax = plt.gca() # Get current axes
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=(num_buckets-1)*bucket_size))
    sm.set_array([]) # You need this line for the colorbar to work with line plots
    cbar = plt.colorbar(sm, ax=ax) # Pass the axes to the colorbar
    cbar.set_label(r'cos($\phi$)')

    if False:
    
        #plot y=0.07x for comparison in green
        x = np.arange(0, 80, 0.1)
        y = 0.026*x+0.46
        plt.plot(x, y/2.2, 'o-', markersize=5, color='green')
        y2 = (0.032*(x))**(0.35)
        plt.plot(x, y2/2, 'o-', markersize=5, color='red')
        y3 = 1 - np.exp(-0.026*((x+1.5)**2))
        plt.plot(x, y3/2.7, 'o-', markersize=5, color='orange')

    #add legend
    #plt.legend([f"Incidence: {i * bucket_size:.1f}" for i in range(num_buckets)])
  
    fig = plt.gcf()
    #fig.set_size_inches(16, 4)
    plt.savefig("intensity_plots/intensity_stats_per_incidence.png")

        #plt.clf()

def plot_intensity_for_incidence_for_distances(bucket_size=4, bin_size=0.1):

    # Calculate the number of buckets needed to cover the range of distances
    num_buckets = int(80 / bucket_size) + 1  # Cover distances up to 80 meters

    # create empty list with the calculated number of elements
    intensity_buckets = [[] for _ in range(num_buckets)]
    incidence_buckets = [[] for _ in range(num_buckets)]

    for i, file in enumerate(files):

        pano = np.load(os.path.join(path, file))
        pano = pano.astype(np.float32)
        #get the intensities
        incidences = pano[:, :, 0]
        #incidences = np.arccos(incidences)/(np.pi/2)
     
        
        distances = pano[:, :, 2]
        intensities = pano[:, :, 1] 
        #distance_buckets cast to int
        distance_buckets = np.round(distances / bucket_size).astype(int)
        

        # Flatten the arrays and filter based on conditions
        valid_indices = (distances.flatten() <= 80) & (intensities.flatten() <= 2) & (incidences.flatten() > 0)  & (incidences.flatten() < .99)
        
        valid_distances = distance_buckets.flatten()[valid_indices]
        valid_intensities = intensities.flatten()[valid_indices]
        valid_incidences = incidences.flatten()[valid_indices]
        
        # Append the valid intensities and incidences to their respective buckets
        for distance in range(num_buckets):
            indices = (valid_distances == distance)
            intensity_buckets[distance].extend(valid_intensities[indices])
            incidence_buckets[distance].extend(valid_incidences[indices])



    k = 0

    for bucket, incidence_bucket in zip(intensity_buckets, incidence_buckets):
        # Calculate mean intensity for each 0.1 incidence bucket
        bucket_intensities = np.array(bucket)


        bucket_incidence = np.array(incidence_bucket)

        if len(bucket_intensities) == 0:
            continue

        # Create incidence buckets of size 0.1
        incidence_bins = np.arange(0, np.max(bucket_incidence) + bin_size, bin_size)
        
        # Calculate mean intensity for each incidence bin
        mean_intensities = []
        incidence_bin_centers = []
        # Vectorize the binning and mean calculation
        bin_indices = np.digitize(bucket_incidence, incidence_bins) - 1  # Subtract 1 to align with array indices
        
        # Filter out-of-bounds indices
        valid_bins = (bin_indices >= 0) & (bin_indices < len(incidence_bins) - 1)
        bin_indices = bin_indices[valid_bins]
        intensities_in_bins = bucket_intensities[valid_bins]

        
        # Calculate the mean intensity for each bin using bincount and groupby
        bin_counts = np.bincount(bin_indices, minlength=len(incidence_bins) - 1)
        bin_sums = np.bincount(bin_indices, weights=intensities_in_bins, minlength=len(incidence_bins) - 1)
        
        # Avoid division by zero
        #set binsumss to 1 if bin counts is 0
        #bin_counts = np.where(bin_counts == 0, 1, bin_counts)
        mean_intensities = np.where(bin_counts > 0, bin_sums / bin_counts, 0)


        #max_intensities = np.where(bin_counts > 0, np.max(intensities_in_bins), 0)
        #mean_intensities = max_intensities
        # Calculate bin centers
        incidence_bin_centers = (incidence_bins[:-1] + incidence_bins[1:]) / 2
        
        # Filter out bins with no data
        valid_bins = bin_counts > 0
        mean_intensities = mean_intensities[valid_bins]
        incidence_bin_centers = incidence_bin_centers[valid_bins]
        
        if len(mean_intensities) == 0:
            continue
        #mean_intensities/=np.max(mean_intensities)*2.1

        
        #compute greyscale based on k and num_buckets
        greyscale = k/num_buckets

        # Plot mean intensities vs incidence bin centers
        #use coolwarm
        plt.plot(incidence_bin_centers, mean_intensities, 'o-', markersize=5, color=plt.cm.coolwarm(greyscale))
        plt.xlabel(r"cos($\phi$)")
        plt.ylabel("Mean Intensity")
        plt.ylim([0, .5])
        #add legend
        #plt.legend([f"Distance: {i * bucket_size:.1f}" for i in range(num_buckets)]) # Removed legend
        k+=1



        # Add colorbar after the loop
    ax = plt.gca() # Get current axes
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=(num_buckets-1)*bucket_size))
    sm.set_array([]) # You need this line for the colorbar to work with line plots
    cbar = plt.colorbar(sm, ax=ax) # Pass the axes to the colorbar
    cbar.set_label('Distance (m)')



    #plot cos(incidence) for comparison
    #incidences = np.arange(0.1, 1, 0.1)
    #cos_incidence = (incidences)**0.1
    #plt.plot(incidences, cos_incidence, 'o-', markersize=5, color='black')
    plt.savefig("intensity_plots/intensity_stats_per_distance.png")

        #plt.clf()
plt.rcParams.update({'font.size': 18})
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlepad'] = 1 # Add this line
plt.rcParams['axes.labelpad'] = 1 # Add this line
    
#plot_intensity_for_distance_for_incidences(bucket_size = 0.1, bin_size =3)
plot_intensity_for_incidence_for_distances(bucket_size = 4, bin_size = 0.04)