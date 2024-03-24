import os
import cv2
import json
import numpy as np
# from screeninfo import get_monitors
import tqdm
import glob

def get_screen_resolution(screen_ind):
    m  = get_monitors()[screen_ind]
    width = m.width
    height = m.height

    return (int(width), int(height))

def create_projector_window(win_name:str, proj_w:int, proj_h:int, pos_x:int, pos_y:int):
    # create projector GUI
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, proj_w, proj_h)
    cv2.moveWindow(win_name, pos_x, pos_y)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)

def create_2x2_grid(imgs, cell_with, cell_height):
    
    img = np.zeros((cell_height*2, cell_with*2,3), dtype=np.uint8)
    if imgs[0] != []:
        img[0:cell_height, 0:cell_with,:] = cv2.resize(imgs[0][:,:,0:3], (cell_with, cell_height))
    if imgs[1] != []:
        img[0:cell_height, cell_with:cell_with*2,:] = cv2.resize(imgs[1][:,:,0:3], (cell_with, cell_height))
    if imgs[2] != []:
        img[cell_height:cell_height*2, 0:cell_with,:] = cv2.resize(imgs[2][:,:,0:3], (cell_with, cell_height))
    if imgs[3] != []:
        img[cell_height:cell_height*2, cell_with:cell_with*2,:] = cv2.resize(imgs[3][:,:,0:3], (cell_with, cell_height))

    return img

def save_params_to_json(filename, data):
    with open(filename, "w") as outfile:
        json_object = json.dumps(data, indent=4)
        outfile.write(json_object)

def load_params_from_json(filename):
    p = []
    
    with open(filename, 'r') as openfile:
        p = json.load(openfile)

    return p

def load_images(data_folder, patterns_nb=None):
    imgs1 = []
    imgs2 = []
    files1 = glob.glob(f'{data_folder}/img1_*.png')
    files2 = glob.glob(f'{data_folder}/img2_*.png')

    # print(patterns_nb, len(files1))
    if patterns_nb == None: patterns_nb = len(files1)

    # for f1, f2 in zip(files1, files2):
    for i in range(0, patterns_nb, 1):
        f1, f2 = files1[i], files2[i]
        if os.path.exists(f1) and os.path.exists(f2):
            imgs1.append(cv2.imread(f1))#.astype(np.float16))
            imgs2.append(cv2.imread(f2))#.astype(np.float16))

    return imgs1, imgs2

def disparity_map_to_color(disparity_map):
    dd = (disparity_map-disparity_map.min())/(disparity_map.max()-disparity_map.min())
    disparity_map_color = cv2.applyColorMap((dd*255).astype(np.uint8), cv2.COLORMAP_JET)

    return disparity_map_color

def remove_points(xyz, rgb, z_lim):
    ind = np.where((xyz[:,2]>z_lim[0]) & (xyz[:,2]<z_lim[1]))[0]
    return xyz[ind,:], rgb[ind,:]