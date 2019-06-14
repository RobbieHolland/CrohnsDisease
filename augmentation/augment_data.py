from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import numpy as np
import cv2
import scipy
import random
import math
import multiprocessing as mp
from multiprocessing import Pool
import functools
import random
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
# from dltk.io.preprocessing import *

max_angle = 10
alpha, sigma = 9e3, 50

def random_displacement(image, lb, up):
    return [round(np.random.uniform(-mx_disp_prop, mx_disp_prop) * d) for d in image.shape]

def crop(image, desired_size, coord, mode='center'):
    diff = image.shape - np.array(desired_size)
    if mode == 'random':
        ds = [random.randint(0, d) for d in diff]
    elif mode == 'center':
        ds = [int(round(d / 2)) for d in diff]

    res = [diff_i - ds_i for diff_i, ds_i in zip(diff, ds)]
    center_displacement = np.array(ds) - np.array(res)
    res = [-r if r != 0 else None for r in res]

    cropped = image[ds[0]:res[0], ds[1]:res[1], ds[2]:res[2]]

    coord = ((coord * image.shape) - center_displacement) / cropped.shape
    return cropped, coord

def random_translate(image):
    displacements = random_displacement(image, -mx_disp_prop, mx_disp_prop)
    return scipy.ndimage.shift(image, displacements, mode='nearest')

def random_rotate(image):
    angle = np.random.uniform(max_angle, max_angle)
    return scipy.ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=5, mode='nearest')

def random_zoom(image):
    zooms = [np.random.uniform(a, b) for a, b in [[1, 1.3], [1, 1.3], [1, 1]]]
    return zoom(image, zooms)

def normalise_image(image):
    return (image - np.min(image)) / (np.max(image))

def random_flip(image, coord, axis=0):
    if np.random.uniform() < 0.5:
        coord[axis] = -coord[axis]
        return np.flip(image, axis), coord
    return image, coord

def augment(im_coord, out_dims=None):
    image, coord = im_coord
    # Initial crop to remove border artefacts
    image = random_rotate(image)
    # image = crop(image, (image.shape[0] - 4, image.shape[1] - 8, image.shape[2] - 8), mode='center')
    # image, coord = random_flip(image, coord, axis=0)

    image, coord = crop(image, (image.shape[0] - 4, image.shape[1] - 8, image.shape[2] - 8), coord, mode='center')
    image, coord = random_flip(image, coord, axis=1)
    image, coord = random_flip(image, coord, axis=2)
    image, coord = crop(image, out_dims, coord, mode='random')

    # image = random_zoom(image)
    # image = crop(image, out_dims, mode='random')
    # image = add_gaussian_noise(image, sigma=0.005)
    # image = elastic_transform(image, alpha=[1, alpha, alpha],
    #                                      sigma=[1 ,sigma, sigma])
    image = normalise_image(image)
    return image, coord

# Process results in the same output shape as augment, and the same standardisation
def parse_test(im_coord, out_dims=None):
    image, coord = im_coord
    image, coord = crop(image, out_dims, coord, mode='center')
    image = normalise_image(image)
    return image, coord

class Augmentor:
    def __init__(self, out_dims):
        self.angle_std = 4
        self.alpha, self.sigma = 5e3, 35
        self.mx_disp_prop = 0.04
        self.mappable_augment = functools.partial(augment, out_dims=out_dims)
        self.mappable_parse_test = functools.partial(parse_test, out_dims=out_dims)

    def __call__(self, image):
        return augment(image, self.out_dims)

    def paralellise_f(self, images_coords, f):
        with Pool(processes=mp.cpu_count()) as pool:
            print(f'Processing {len(images_coords)} images \r', end='')
            result = pool.map(f, images_coords)
            result = np.array(result)
            augmented, coords = result[:,0], result[:,1]
            return np.stack(augmented), np.stack(coords)

    def augment_batch(self, images, coords):
        return self.paralellise_f(list(zip(images, coords)), self.mappable_augment)

    def parse_test_features(self, images, coords):
        return self.paralellise_f(list(zip(images, coords)), self.mappable_parse_test)
