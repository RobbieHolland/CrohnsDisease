import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data.handling.data_handler import DataHandler

logdir = '/vol/bitbucket/rh2515/CrohnsDisease/eda_logdir'


data_handler = DataHandler('/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY', './data/cases/index')
healthy, healthy_labels, abnormal, abnormal_labels = data_handler.load_images()

def show_data(images, title):
    figure_size = (12, 9)
    fig = plt.figure(figsize=figure_size)
    fig.suptitle(title)

    # View dataset images
    for i, image in enumerate(images):
        fig.add_subplot(figure_size[0], figure_size[1], i + 1)
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    plt.show()

print('Display healthy slices')
show_data(healthy, 'Healthy slices')
print('Display abnormal slices')
show_data(abnormal, 'Abnormal slices')
