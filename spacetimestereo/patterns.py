import numpy as np
import cv2

def create_one_line_pattern(patterns_nb, intensity, width, img_size, blur_size=0):

    patterns = []
    for i in range(patterns_nb):
        img = np.zeros(img_size, dtype=np.uint8)
        for w in range(width):
            img[:,i+w:-1:patterns_nb] = intensity
        if blur_size>0:
            img = cv2.GaussianBlur(img, (blur_size,blur_size), 0)
        patterns.append(img)
    return patterns

def create_one_line_random_pattern(patterns_nb, intensity, img_size):
    
    patterns = []
    ind = np.random.permutation(img_size[1])
    for i in range(patterns_nb):
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        img[:,ind[i:-1:patterns_nb]] = intensity
        patterns.append(img)
        
    return patterns

def create_one_line_blurred_random_pattern(patterns_nb, intensity, img_size, blur_size=7):
    
    patterns = []
    ind = np.random.permutation(img_size[1])
    for i in range(patterns_nb):
        img = np.zeros(img_size, dtype=np.uint8)
        img[:,ind[i:-1:patterns_nb]] = intensity
        img = cv2.GaussianBlur(img, (blur_size,blur_size), 0)
        patterns.append(img)
    return patterns
    

def create_black_white_pattern(patterns_nb, intensity, img_size):
    
    patterns = [] 
    for i in range(patterns_nb):
        k = i/(patterns_nb-1)
        img = np.zeros(img_size, dtype=np.uint8) + int(intensity*k)
        patterns.append(img)

    return patterns