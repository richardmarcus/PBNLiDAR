

from model.runner import plot_distance_normalization




#plot_distance_normalization(near_range_threshold=14,near_range_factor=0.42, scale=0.01, near_offset = 2.2, distance_fall = 5.0)
# 
#plot_distance_normalization(near_range_threshold=14,near_range_factor=0.42, scale=0.01, near_offset = 2.2, distance_fall = 1.0)

'''near_range_threshold tensor(13.3791, device='cuda:0', requires_grad=True)
loaded near_range_factor to opt
near_range_factor tensor(0.2381, device='cuda:0', requires_grad=True)
loaded distance_scale to opt
distance_scale tensor(0.0051, device='cuda:0', requires_grad=True)
loaded near_offset to opt
near_offset tensor(2.2770, device='cuda:0', requires_grad=True)
loaded distance_fall to opt
distance_fall tensor(4.2094, device='cuda:0', requires_grad=True)'''


#plot_distance_normalization(near_range_threshold=13.3791,near_range_factor=0.2381, scale=0.0051, near_offset = 2.2770, distance_fall = 4.2094)

'''        opt.distance_scale = torch.tensor(0.01).to(device)
        opt.near_range_threshold = torch.tensor(14.0).to(device)
        opt.near_range_factor = torch.tensor(0.45).to(device)
        opt.near_offset = torch.tensor(2.0).to(device)
        opt.distance_fall = torch.tensor(5.0).to(device)
'''
plot_distance_normalization(near_range_threshold=-14.0,near_range_factor=0.45, scale=0.0, near_offset = 2.0, distance_fall = 1.0)