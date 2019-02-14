from dltk.io.augmentation import *
import numpy as np
# from dltk.io.preprocessing import *

class Augmentor:
    def __init__(self):
        pass

    def augment(self, images):
        alpha, sigma = 3e2, 25
        for i in range(len(images)):
            image = flip(images[i], axis=1)
            images = add_gaussian_noise(images, sigma=0.005)
            images[i] = elastic_transform(image, alpha=[alpha, alpha], sigma=[sigma, sigma])
        return images
