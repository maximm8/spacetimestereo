import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

import spacetimestereo as sts


if __name__ == '__main__':
    
    data_folder = f'results/candle/'
    print('dataset:', data_folder)
    
    # load results
    disparity_map = np.load(f'{data_folder}/disparity.npy', allow_pickle=True)
    pcd = o3d.io.read_point_cloud(f"{data_folder}point_cloud.ply")

     # show disparity map
    plt.imshow(disparity_map)
    plt.show()

    # show point cloud
    o3d.visualization.draw_geometries([pcd])

