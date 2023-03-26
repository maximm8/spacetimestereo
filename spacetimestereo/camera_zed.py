import cv2
import numpy as np
import pyzed.sl as zedsl

from spacetimestereo.camera import *

class camera_zed(camera):
    def __init__(self):
        # Create a ZED camera object
        self.zed = zedsl.Camera()

    def __del__(self):
        self.zed.close()

    def set_params(self, img_width:int, fps=15, exposure=50):
        # Set configuration parameters
        init_params = zedsl.InitParameters()

        self.zedres = zedsl.Resolution()

        if img_width == 2208:
            init_params.camera_resolution = zedsl.RESOLUTION.HD2K
            self.zedres.width  = 2208
            self.zedres.height = 1242
        elif img_width == 1920:
            init_params.camera_resolution = zedsl.RESOLUTION.HD1080        
            self.zedres.width  = 1920
            self.zedres.height = 1080
        elif img_width == 1280:
            init_params.camera_resolution = zedsl.RESOLUTION.HD720
            self.zedres.width  = 1280
            self.zedres.height = 720
        else:
            init_params.camera_resolution = zedsl.RESOLUTION.VGA 
            self.zedres.width  = 672
            self.zedres.height = 376

        # init_params.depth_mode = zedsl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        # init_params.depth_mode = zedsl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode
        # init_params.depth_mode = zedsl.DEPTH_MODE.NONE  # Use PERFORMANCE depth mode
        # init_params.depth_mode = zedsl.DEPTH_MODE.QUALITY  # Use PERFORMANCE depth mode
        init_params.depth_mode = zedsl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode

        init_params.coordinate_units = zedsl.UNIT.METER  # Use meter units (for depth measurements)
        
        init_params.camera_fps = fps
        init_params.depth_maximum_distance = 2.0
        init_params.depth_minimum_distance = 0.1

        # Set sensing mode in FILL
        runtime_parameters = zedsl.RuntimeParameters()
        runtime_parameters.sensing_mode = zedsl.SENSING_MODE.FILL

        # Open the camera
        err = self.zed.open(init_params)
        if (err != zedsl.ERROR_CODE.SUCCESS) :
            return -1

        # Set exposure  and white balance
        self.zed.set_camera_settings(zedsl.VIDEO_SETTINGS.EXPOSURE, exposure)        
        self.zed.set_camera_settings(zedsl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)

        self.image1 = zedsl.Mat()
        self.image2 = zedsl.Mat()
        self.image_depth = zedsl.Mat()
        self.point_cloud = zedsl.Mat()
        self.disparity_map = zedsl.Mat()
        
        return 0

    def grab(self):
        if (self.zed.grab() == zedsl.ERROR_CODE.SUCCESS) :
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(self.image1, zedsl.VIEW.LEFT) # Get the left image
            self.zed.retrieve_image(self.image2, zedsl.VIEW.RIGHT) # Get the right image
            self.zed.retrieve_image(self.image_depth, zedsl.VIEW.DEPTH)
            self.zed.retrieve_measure(self.point_cloud, zedsl.MEASURE.XYZRGBA, zedsl.MEM.CPU)#, self.zedres)
            self.zed.retrieve_measure(self.disparity_map, zedsl.MEASURE.DISPARITY)

            return 0

        return -1

    def get_data(self, img_type:image_type):
        data = []

        if img_type == image_type.LEFT_IMG:
            data = self.image1.get_data()
        elif img_type == image_type.RIGHT_IMG:
            data = self.image2.get_data()
        elif img_type == image_type.DISPARITY_MAP:            
            data = self.disparity_map.get_data()
        elif img_type == image_type.POINT_CLOUD:
            data = self.point_cloud.get_data()
        elif img_type == image_type.DEPTH_IMG:
            data = self.image_depth.get_data()
        
        return data
    
    def get_img_size(self):
        return (self.zedres.width, self.zedres.height)

    def get_camera_params(self):
        cam_params = {}
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        intrinsics = [calibration_params.left_cam, calibration_params.right_cam]

        for k, intr in enumerate(intrinsics):
            intr_params = {'fx':intr.fx, 'fy':intr.fy, 'dist': list(intr.disto), 'ppx':intr.cx, 'ppy':intr.cy, 'w':intr.image_size.width, 'h':intr.image_size.height}
            cam_params[f'intr{k}'] = intr_params

        rmat2, _ = cv2.Rodrigues(np.array(calibration_params.R))
        tvec2 = np.array(calibration_params.T)
        rmat1 = np.linalg.inv(rmat2)
        tvec1 = np.matmul(-np.linalg.inv(rmat2),tvec2)

        extr_params = {'rotation':list(rmat1.ravel()), 'translation':list(tvec1.ravel())}
        cam_params['extr0'] = extr_params
        extr_params = {'rotation':list(rmat2.ravel()), 'translation':list(tvec2.ravel())}
        cam_params['extr1'] = extr_params

        return cam_params

    def set_exposure(self, exp_value):
        self.zed.set_camera_settings(zedsl.VIDEO_SETTINGS.EXPOSURE, exp_value)

    def get_exposure(self):
        return self.zed.get_camera_settings(zedsl.VIDEO_SETTINGS.EXPOSURE)


    



