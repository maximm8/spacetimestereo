import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    
    result_folder = f'results/'

    dataset_name = 'candle'
    # dataset_name = 'lego'

    data_folder = f'{result_folder}/{dataset_name}/'

    print('dataset:', dataset_name)

    # load results
    disparity_map = np.load(f'{data_folder}/disparity.npy', allow_pickle=True)
    pcd = o3d.io.read_point_cloud(f"{data_folder}point_cloud.ply")

     # show disparity map
    plt.imshow(disparity_map)
    plt.show()

    # show point cloud
    o3d.visualization.draw_geometries([pcd])