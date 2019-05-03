import tensorflow as tf
from scripts.show import *
from pipeline import Pipeline
from augmentation.augment_data import *
from main_util import generate_decode_function

train_data = '/vol/bitbucket/rh2515/CrohnsDisease/Crohns/tfrecords/axial_t2_only_train.tfrecords'
test_data = '/vol/bitbucket/rh2515/CrohnsDisease/Crohns/tfrecords/axial_t2_only_test.tfrecords'
feature_shape=(64, 128, 256)

# Dataset pipeline
decode_record = generate_decode_function(feature_shape, 'axial_t2')
pipeline = Pipeline(decode_record, train_data, test_data)
iterator, _ = pipeline.create(volume_shape=feature_shape, batch_size=10, test_size=10)
features, labels = iterator.get_next()

# Augmentation
augmentor = Augmentor()

with tf.Session() as sess:
    # Initialise
    sess.run(iterator.initializer)

    batch_images = sess.run(features)
    aug_batch_images = augmentor.augment_batch(np.copy(batch_images))

    a = 0
    for vol, aug_vol in zip(batch_images, aug_batch_images):
        slice = 30
        img, aug_img = vol[slice], aug_vol[slice]
        fig=plt.figure(figsize=(1, 2))
        fig.set_size_inches(10, 5)

        fig.add_subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        fig.add_subplot(1, 2, 2)
        plt.imshow(aug_img, cmap='gray')
        plt.title('Augmented')
        plt.savefig(f'images/foo{a}.png')
        a += 1
