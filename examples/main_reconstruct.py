import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

import spacetimestereo as sts


if __name__ == '__main__':

    dataset_folder = f'data/'
    result_folder = f'results/'

    dataset_name = 'candle'
    dataset_name = 'lego'

    data_folder = f'{dataset_folder}/{dataset_name}/'
    output_folder = f'{result_folder}/{dataset_name}/'

    print('dataset:', dataset_name)

    shadow_th           = 50
    black_white_ind     = [0, 1]
    disp_range          = (400, 700)
    device              = 'cuda'# or 'cpu'
    batches             = 4 # try 10 if out of memory
    filt_size           = 3 # spatial smoothing window size

    ss  = sts.SpacetimeStereo()

    # load data
    ss.load_params(data_folder)
    imgs1, imgs2 = ss.load_images(data_folder)
    black_img, white_img = imgs1[black_white_ind[0]], imgs1[black_white_ind[1]]

    # calculate disparity map using spacetime stereo
    disparity_map = ss.calc_disparity(imgs1, imgs2, disp_range,filt_size, device, batches)
    shadow_map = ss.calc_shadow_map_from_bw(black_img, white_img, shadow_th)
    disparity_map *= shadow_map

    pcd = ss.disparity_to_point_cloud(disparity_map, imgs1[black_white_ind[1]], z_lim=[0, 2])

    # postprocessing point cloud using open3D
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.001)

    # show disparity map
    plt.imshow(disparity_map)
    plt.show()

    #show point cloud
    o3d.visualization.draw_geometries([pcd])

    # save results
    # if not os.path.exists(output_folder): os.mkdir(output_folder)
    # disparity_map_color =  sts.disparity_map_to_color(disparity_map)
    # cv2.imwrite(f'{output_folder}/disparity.png', disparity_map_color)
    # np.save(f'{output_folder}/disparity.npy', disparity_map)
    # o3d.io.write_point_cloud(f'{output_folder}point_cloud.ply', pcd)