import pydicom
import matplotlib.pyplot as plt
import tensorflow as tf
from data.data_handler import DataHandler
from model.vgg import Model

# Hyperparameters
portion_train = 0.8
batch_size = 16
image_width = 512
image_height = 512

# Load data into memory (dataset small)
data_handler = DataHandler('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
features, labels = data_handler.load_dataset()

ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.shuffle(len(features))
ds = ds.batch(batch_size)
ds_iter = ds.make_initializable_iterator()
next_batch = ds_iter.get_next()

print('Loaded images into memory')

# Initialise Model
network = Model(image_width, image_height, batch_size)

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(ds_iter.initializer)
        batch = sess.run(next_batch)
