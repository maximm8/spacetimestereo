import numpy as np
import json
import cv2
import os

import spacetimestereo as sts

def get_acquisition_params(cam, proj_resolution, patterns_nb, black_white_img_ind):
    
    acq_params = {  'cam_width':cam.get_img_size()[0], 
                    'cam_height:':cam.get_img_size()[1], 
                    'proj_width':proj_resolution[0], 
                    'proj_height':proj_resolution[1],
                    'patterns_nb':patterns_nb, 
                    'pattern_black_ind':black_white_img_ind[0], 
                    'pattern_white_ind':black_white_img_ind[1]}

    return acq_params

if __name__ == '__main__':
    
    data_folder = 'data/'

    line_width = 1
    patterns_nb = 50
    line_intensity = 128
    white_bkg_intensity = 128

    capture = False
    pattern_ind = 0
    exp_step = 5
    imgs1, imgs2, disps, pcs = [], [], [], []

    # create camera
    cam = sts.camera_zed()
    cam.set_params(img_width=1920, fps=30, exposure=50)

    # get screen resolutions
    screen_resolution = sts.get_screen_resolution(1)
    proj_resolution = sts.get_screen_resolution(0)
    
    ss = sts.SpacetimeStereo()
    
    # create patterns
    patterns = ss.generate_pattern(patterns_nb, line_intensity, white_bkg_intensity, proj_resolution)

    # create projector window
    sts.create_projector_window("projector", proj_resolution[0], proj_resolution[1], screen_resolution[0]+1, 0)

    print('-------------------------------------')
    print('press space to start/stop image aquisition')
    print('press e/d to adjust exposure')
    print('press w/s to change pattern')
    print('-------------------------------------')
    
    while True:
        cv2.imshow('projector', patterns[pattern_ind])

        if capture:
            cv2.waitKey(250) 

        if cam.grab() == 0:

            img1        = cam.get_data(sts.image_type.LEFT_IMG)
            img2        = cam.get_data(sts.image_type.RIGHT_IMG)
            img_depth   = cam.get_data(sts.image_type.DEPTH_IMG)
            disparity   = cam.get_data(sts.image_type.DISPARITY_MAP)
            point_cloud = cam.get_data(sts.image_type.POINT_CLOUD)

            image_depth_color = cv2.applyColorMap(img_depth[:,:,0], cv2.COLORMAP_JET)
            disparity_color = cv2.applyColorMap((-disparity/1000*255).astype(np.uint8), cv2.COLORMAP_JET)
            
            img = sts.create_2x2_grid([img1, img2, disparity_color, img_depth[:,:,0:3]], 960, 512) 
            cv2.imshow('img', img)

        key = cv2.waitKey(1)
        
        if key == 32:
            capture = not capture
        elif key == 119:#w next pattern 
            pattern_ind = (pattern_ind+1)% len(patterns)
        elif key == 115:#s prev pattern
            pattern_ind = (pattern_ind-1)% len(patterns)
        if key == 101:#e increase exposure
            exp = cam.get_exposure()            
            cam.set_exposure(min(100,exp+exp_step))            
            exp_new = cam.get_exposure()
            print(f"cam: {cam} set exposure:", exp_new)
        elif key == 100:#d decrease exposure
            exp = cam.get_exposure()
            cam.set_exposure(max(1,exp-exp_step))
            exp_new = cam.get_exposure()
            print(f"cam: {cam} set exposure:", exp_new)            
        elif key == 27:
            break

        if capture:
            imgs1.append(img1.copy())
            imgs2.append(img2.copy()) 
            disps.append(disparity.copy())
            pcs.append(point_cloud.copy())
            pattern_ind += 1
            print(f'captured frame:{len(imgs1)}/{len(patterns)}')    

        if pattern_ind>= len(patterns):
            break

    # save sequence params
    print('saving params')
    cam_params = cam.get_camera_params()
    acq_params = get_acquisition_params(cam, proj_resolution, len(patterns), (0, 1))
    ss.save_params(data_folder, cam_params, acq_params)

    # save captured data
    print('saving images')
    for i in range(len(imgs1)):
        cv2.imwrite(f'{data_folder}/img1_{i:02d}.png', imgs1[i])
        cv2.imwrite(f'{data_folder}/img2_{i:02d}.png', imgs2[i])
        cv2.imwrite(f'{data_folder}/pattern_{i:02d}.png', patterns[i])
        
        #save point cloud and dispary map captured by camera
        if i<=acq_params["pattern_white_ind"]:
            np.save(f'{data_folder}/disparity_{i:02d}.npy', disps[i])
            np.save(f'{data_folder}/pc_{i:02d}.npy', pcs[i])