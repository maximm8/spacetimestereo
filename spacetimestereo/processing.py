import cv2
import numpy as np
import spacetimestereo.utils as tools
import matplotlib.pyplot as plt
import open3d as o3d

def calc_shadow_map(black, white, th):
    shadow_map = np.zeros_like(black)
    d = white.astype(np.float32)-black.astype(np.float32)
    ind = np.where(d>th)
    if len(shadow_map.shape) == 2:
        shadow_map[ind[0], ind[1]] = 1
    else:
        shadow_map[ind[0], ind[1], ind[2]] = 1

    shadow_map= shadow_map.astype(np.uint8)
    shadow_map[:,:, 0] = shadow_map[:,:,0]|shadow_map[:,:,1]|shadow_map[:,:,2]

    return shadow_map[:,:,0]

def load_disparity_map(filename):
    
    file = open(filename,"rb")

    w = int.from_bytes(file.read(4), byteorder='little')
    h = int.from_bytes(file.read(4), byteorder='little')
    

    img_data = file.read()
    file.close()

    disparity_map = np.frombuffer(img_data, dtype='int')
    disparity_map = np.reshape(disparity_map, (h, w)).T

    # disparity_map = img_data.reshape((w, h))

    return disparity_map

def disparity_to_point_cloud(disparity_map, point_colors, Q):

    image_size = disparity_map.shape
    point_cloud = np.zeros((image_size[0], image_size[1],4))
    disparity_map = disparity_map.astype(np.float32)
    point_cloud = cv2.reprojectImageTo3D(disparity_map, Q, point_cloud, True, -1)
    x = point_cloud[:,:,0].reshape(-1,1)
    y = point_cloud[:,:,1].reshape(-1,1)
    z = point_cloud[:,:,2].reshape(-1,1)
    c = point_colors.reshape(-1,3)
    # ind = np.where(z<500)[0]
    # xyz = np.concatenate((x[ind],y[ind],z[ind]), axis=1)
    xyz = np.concatenate((x,y,z), axis=1)
    rgb = np.zeros_like(xyz)
    # rgb[:,0] = c[ind,2]
    # rgb[:,1] = c[ind,1]
    # rgb[:,2] = c[ind,0]

    rgb[:,0] = c[:,2]
    rgb[:,1] = c[:,1]
    rgb[:,2] = c[:,0]

    return xyz, rgb

def point_cloud_to_xyz(point_cloud):
    x = point_cloud[:,:,0].reshape(-1, 1)
    y = point_cloud[:,:,1].reshape(-1, 1)
    z = point_cloud[:,:,2].reshape(-1, 1)
    xyz = np.concatenate((x,y,z), axis=1)
    return xyz

def float_to_rgb(f):
    
    s = struct.pack('>f', f)
    bits = struct.unpack('>l', s)[0]
    r = bits & 0xFF
    g = bits >> 8 & 0xFF
    b = bits >> 16 & 0xFF
    return (r,g,b)

def array2d_to_rgb(f):
    bytes_arr = f.tobytes()
    rgb  = np.frombuffer(bytes_arr, dtype=np.uint8)
    rgb = rgb.reshape(f.shape[0], f.shape[1], 4)
    return rgb

def point_cloud_zed_to_point_cloud(point_cloud_zed):
    xyz2 = point_cloud_to_xyz(point_cloud_zed)
    c2 = array2d_to_rgb(point_cloud_zed[:,:,3])
    r = c2[:,:,0].reshape(-1, 1)
    g = c2[:,:,1].reshape(-1, 1)
    b = c2[:,:,2].reshape(-1, 1)
    rgb = np.concatenate((r,g,b), axis=1)

def create_point_cloud_open3d(xyz, c):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float32)/255)
    return pcd

def extract_params(cam_params, cam_ind:int):
    intr_str = f'intr{cam_ind}'
    extr_str = f'extr{cam_ind}'
    K = np.array([[cam_params[intr_str]['fx'], 0, cam_params[intr_str]['ppx']], 
                [0, cam_params[intr_str]['fy'], cam_params[intr_str]['ppy']], 
                [0, 0, 1]])

    distort = np.array(cam_params[intr_str]['dist'])

    rmat = np.array(cam_params[extr_str]['rotation']).reshape(3,3)
    tvec = np.array(cam_params[extr_str]['translation'])

    return K, distort, rmat, tvec


def create_reporjection_matrix(cam_params, image_size):
    
    K1, dist1, rmat1, tvec1 = extract_params(cam_params, 0)
    K2, dist2, rmat2, tvec2 = extract_params(cam_params, 1)

    R1 = np.zeros((3,3))
    R2 = np.zeros((3,3))
    P1 = np.zeros((3,4))
    P2 = np.zeros((3,4))
    Q1 = np.zeros((4,4))
    Q2 = np.zeros((4,4))

    R1, R2, P1, P2, Q1, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, dist1, K2, dist2, image_size, rmat1, tvec1, R1, R2, P1, P2, Q1, 0, -1)
    R1, R2, P1, P2, Q2, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, dist1, K2, dist2, image_size, rmat2, tvec2, R1, R2, P1, P2, Q2, 0, -1)

    return Q1, Q2


