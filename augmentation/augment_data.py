from dltk.io.augmentation import *
import numpy as np
import cv2
import scipy
import random
import math
# from dltk.io.preprocessing import *

class Augmentor:
    def __init__(self):
        self.angle_std = 4
        self.alpha, self.sigma = 5e3, 35
        self.mx_disp_prop = 0.04

    def random_translate(self, image):
        x, y, z = [round(np.random.uniform(-self.mx_disp_prop, self.mx_disp_prop) * d) for d in image.shape]
        return scipy.ndimage.shift(image, [x, y, z], mode='nearest')

    def random_rotate(self, image):
        angle = np.random.normal(loc=0, scale=self.angle_std)
        return scipy.ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=5, mode='nearest')

    def augment_batch(self, images):
        print(f'Augmenting {len(images)} images')
        for i in range(len(images)):
            print(f'Augmenting image {i + 1}\r', end='')
            images[i] = self.augment(images[i])
        return images

    def augment(self, image):
        image = flip(image, axis=2)
        image = self.random_rotate(image)
        image = self.random_translate(image)
        image = add_gaussian_noise(image, sigma=0.005)
        # image = elastic_transform(image, alpha=[self.alpha, self.alpha, self.alpha],
        #                                      sigma=[self.sigma ,self.sigma, self.sigma])
        return image
