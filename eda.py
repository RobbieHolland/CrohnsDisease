import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data.data_handler import DataHandler

logdir = '/vol/bitbucket/rh2515/CrohnsDisease/eda_logdir'


data_handler = DataHandler('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
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

show_data(healthy, 'Healthy slices')
show_data(abnormal, 'Abnormal slices')
