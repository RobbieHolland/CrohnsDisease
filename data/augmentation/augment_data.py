from dltk.io.augmentation import *
import numpy as np
# from dltk.io.preprocessing import *

class Augmentor:
    def __init__(self):
        pass

    def augment(self, images):
        images = flip(images, axis=1)
        images = flip(images, axis=0)
        images = add_gaussian_noise(images, sigma=0.1)
        images = elastic_transform(images, alpha=[1e4, 1e4], sigma=[120, 120])
        return images
