from enum import Enum

class image_type(Enum):
    LEFT_IMG        = 1 
    RIGHT_IMG       = 2
    POINT_CLOUD     = 3
    DISPARITY_MAP   = 4
    DEPTH_IMG       = 5

class camera():
    def set_params(self, image_width:int):
        raise NotImplementedError()

    def grab(self):
        raise NotImplementedError()

    def get_data(self, img_type:image_type):
        raise NotImplementedError()

    def set_exposure(self, exp_value:int):
        raise NotImplementedError()

    def get_exposure(self):
        raise NotImplementedError()



