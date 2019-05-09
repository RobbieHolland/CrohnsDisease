from dltk.io.augmentation import *
import numpy as np
import cv2
import scipy
import random
import math
import multiprocessing as mp
from multiprocessing import Pool
# from dltk.io.preprocessing import *

angle_std = 4
alpha, sigma = 5e3, 35
mx_disp_prop = 0.04

def random_displacement(image, lb, up):
    return [round(np.random.uniform(-mx_disp_prop, mx_disp_prop) * d) for d in image.shape]

################## Not yet tested
def random_crop(image, desired_size):
    dim = image.shape
    ds = random_displacement(image, -mx_disp_prop, mx_disp_prop)
    ds_max = random_displacement(image, mx_disp_prop, mx_disp_prop)
    return image[ds[0]:dim[0] - ds_max[0]][ds[1]:dim[1] - ds_max[1]][ds[1]:dim[1] - ds_max[1]]

def random_translate(image):
    displacements = random_displacement(image, -mx_disp_prop, mx_disp_prop)
    return scipy.ndimage.shift(image, displacements, mode='nearest')

def random_rotate(image):
    angle = np.random.normal(loc=0, scale=angle_std)
    return scipy.ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=5, mode='nearest')

def augment(image):
    image = flip(image, axis=2)
    image = random_rotate(image)
    image = random_translate(image)
    image = add_gaussian_noise(image, sigma=0.005)
    image = elastic_transform(image, alpha=[alpha, alpha, alpha],
                                         sigma=[sigma ,sigma, sigma])
    return image

class Augmentor:
    def __init__(self):
        self.angle_std = 4
        self.alpha, self.sigma = 5e3, 35
        self.mx_disp_prop = 0.04

    def augment_batch(self, images):
        with Pool(processes=mp.cpu_count()) as pool:
            print(f'Augmenting {len(images)} images')
            return pool.map(augment, images)
