import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import struct
import math

from tqdm import tqdm

from spacetimestereo.patterns import *
from spacetimestereo.processing import *
from spacetimestereo.utils import *


class SpacetimeStereo:

    def __init__(self):
        self.cam_params = []
        self.acq_params = []

    def load_params(self, data_folder):
        self.cam_params = load_params_from_json(f'{data_folder}camera_params.json')
        self.acq_params = load_params_from_json(f'{data_folder}acquisition_params.json')

    def load_images(self, data_folder):
        imgs1, imgs2 = load_images(data_folder, self.acq_params['patterns_nb'])
        return imgs1, imgs2

    @staticmethod
    def save_params(data_folder, cam_params, acq_params):
        save_params_to_json(f'{data_folder}camera_params.json', cam_params)
        save_params_to_json(f'{data_folder}acquisition_params.json', acq_params)

    @staticmethod
    def generate_pattern(patterns_nb, line_intensity, bkg_intensity, img_size):

        p0 = create_black_white_pattern(2, bkg_intensity, img_size)
        p1 = create_one_line_random_pattern(patterns_nb, line_intensity, img_size)
        patterns = []
        patterns.extend(p0)
        patterns.extend(p1)
        return patterns

    @staticmethod
    def calc_disparity(imgs1, imgs2, black_white_ind, disp_range, shadow_th, device='', batches=4, filter_size=3):

        black_ind = black_white_ind[0]
        white_ind = black_white_ind[1]
        start_ind = white_ind+1

        # N = len(imgs1)
        h, w, ch = imgs1[0].shape
        image_size = imgs1[0].shape[0],  imgs1[0].shape[1] 
        

        imgs1 = np.array(imgs1).astype(np.float16)
        imgs2 = np.array(imgs2).astype(np.float16)
        
        shadow_map = calc_shadow_map(imgs1[black_ind], imgs1[white_ind], shadow_th)    
        shadow_map = cv2.medianBlur(shadow_map, 5)

        imgs1 = imgs1.transpose((0,3,1,2))
        imgs2 = imgs2.transpose((0,3,1,2))
        disp_map_diff = np.zeros((disp_range[1]-disp_range[0], h, w))+0000
        
        if device == '':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # calc cost for different disparity values
        y_step= math.ceil(h/batches)
        for y in range(0, h, y_step):
            y_max = min(y+y_step, h)
            imgs1t = torch.tensor(imgs1[start_ind:,:,y:y_max,:], dtype=torch.float16).to(device)
            imgs2t = torch.tensor(imgs2[start_ind:,:,y:y_max,:], dtype=torch.float16).to(device)

            loop = tqdm.tqdm(range(disp_range[0], disp_range[1], 1), leave=False)
            for i, d in enumerate(loop):
            
                t = imgs2t[:,:,:,0:w-d] - imgs1t[:,:,:,d:w]            
                diff = torch.sqrt(torch.sum(torch.mean(t**2, axis=0), axis=0))
                disp_map_diff[d-disp_range[0],y:y_max,d:w] = diff.to('cpu').numpy()            

                loop.set_description(f'calc disparity {int(y/y_step+1)}/{batches}')

            torch.cuda.empty_cache()

        # apply spatial averaging
        if filter_size>1:
            print(f'apply spatial filtering with windows size:{filter_size}')
            
            disp_map_difft = torch.tensor(disp_map_diff, dtype=torch.float16, device=device)
            disp_map_difft = torch.unsqueeze(disp_map_difft, 1)

            filtert = torch.tensor(np.ones((1,1,filter_size,filter_size), dtype=np.float32)/(filter_size*filter_size), 
                                            dtype=torch.float16, device=device)
            disp_map_difft = torch.nn.functional.conv2d(disp_map_difft, filtert, padding=int(filter_size/2))
            disp_map_diff = np.squeeze(disp_map_difft.to('cpu').numpy())
        
        # calc disparity
        disp_min_ind = np.argmin(disp_map_diff, axis=0)
        disp_min_ind = cv2.medianBlur(disp_min_ind.astype(np.float32), 5)    
        disparity_map = -disp_min_ind-disp_range[0]
        
        # remove shadow area
        disparity_map *= shadow_map

        return disparity_map

    def disparity_to_point_cloud(self, disparity_map, texture_img):
        # transform disparity map to point cloud
        image_size = texture_img.shape[0],  texture_img.shape[1] 
        Q, Q2  = create_reporjection_matrix(self.cam_params, image_size)
        xyz, c = disparity_to_point_cloud(disparity_map, texture_img, Q2) 
        pcd = creat_point_cloud_open3d(xyz, c)    

        return pcd
    
