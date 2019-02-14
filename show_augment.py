import tensorflow as tf
from scripts.show import *
from pipeline import Pipeline
from data.augmentation.augment_data import *

train_data = 'data/tfrecords/train_all.tfrecords'
test_data = 'data/tfrecords/test_all.tfrecords'

# Dataset pipeline
pipeline = Pipeline(train_data, test_data)
iterator, _ = pipeline.create()
features, labels = iterator.get_next()

# Augmentation
augmentor = Augmentor()

with tf.Session() as sess:
    # Initialise
    sess.run(iterator.initializer)

    batch = 0
    while (True):
        batch_images = sess.run(features)
        aug_batch_images = augmentor.augment(np.copy(batch_images))

        for img, aug_img in zip(batch_images, aug_batch_images):
            fig=plt.figure(figsize=(1, 2))
            fig.set_size_inches(10, 5)

            fig.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            fig.add_subplot(1, 2, 2)
            plt.imshow(aug_img, cmap='gray')
            plt.show()
