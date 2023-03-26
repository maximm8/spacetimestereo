import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

import spacetimestereo as sts


if __name__ == '__main__':

    data_folder = f'data/candle/'
    data_folder_out = f'results/candle/'
    # data_folder = f'data/lego/'
    # data_folder_out = f'results/lego/'

    print('dataset:', data_folder)

    shadow_th = 50
    black_white_ind = (0, 1)
    disp_range = (400, 700)
    device = 'cuda'# or 'cpu'
    batches = 4 # try 10 if out of memory
    filt_size = 3 # spatial smoothing window size  

    ss  = sts.SpacetimeStereo()

    # load data
    ss.load_params(data_folder)
    imgs1, imgs2 = ss.load_images(data_folder)
    
    # calculate disparity map using spacetime stereo
    disparity_map = ss.calc_disparity(imgs1, imgs2, black_white_ind, disp_range, shadow_th, device, batches, filt_size)
    pcd = ss.disparity_to_point_cloud(disparity_map, imgs1[black_white_ind[1]])

    # postprocessing point cloud using open3D
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.001)

    # show disparity map
    plt.imshow(disparity_map)
    plt.show()

    #show point cloud
    o3d.visualization.draw_geometries([pcd])
    
    # save results
    if not os.path.exists(data_folder_out): os.mkdir(data_folder_out)
    disparity_map_color =  sts.disparity_map_to_color(disparity_map)
    cv2.imwrite(f'{data_folder_out}/disparity.png', disparity_map_color)
    np.save(f'{data_folder_out}/disparity.npy', disparity_map)
    o3d.io.write_point_cloud(f'{data_folder_out}point_cloud.ply', pcd)