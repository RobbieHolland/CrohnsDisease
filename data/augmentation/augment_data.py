from dltk.io.augmentation import *
import numpy as np
import cv2
import scipy
# from dltk.io.preprocessing import *

class Augmentor:
    def __init__(self):
        self.angle_std = 10
        self.alpha, self.sigma = 6e3, 25

    def random_rotate(self, image):
        angle = np.random.normal(loc=0, scale=self.angle_std)
        return scipy.ndimage.interpolation.rotate(image, angle, axes=(1, 2), reshape=False, order=5, mode='nearest')

    def augment_batch(self, images):
        for i in range(len(images)):
            images[i] = self.augment(images[i])
        return images

    def augment(self, image):
        image = flip(image, axis=2)
        # image = self.random_rotate(image)
        image = add_gaussian_noise(image, sigma=0.005)
        image = elastic_transform(image, alpha=[self.alpha, self.alpha, self.alpha],
                                             sigma=[self.sigma ,self.sigma, self.sigma])
        return image
