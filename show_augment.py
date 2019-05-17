import tensorflow as tf
from scripts.show import *
from pipeline import Pipeline
from augmentation.augment_data import *
from main_util import generate_decode_function

train_data = '/vol/gpudata/rh2515/MRI_Crohns/tfrecords/cropped/axial_t2_only_cropped_train_fold4.tfrecords'
test_data = '/vol/gpudata/rh2515/MRI_Crohns/tfrecords/cropped/axial_t2_only_cropped_test_fold4.tfrecords'
feature_shape=(60,132,300)

# Dataset pipeline
decode_record = generate_decode_function(feature_shape, 'axial_t2')
pipeline = Pipeline(decode_record, train_data, test_data)
iterator, iterator_te = pipeline.create(volume_shape=feature_shape, batch_size=48, test_size=12)
features, labels, levels = iterator.get_next()

# Augmentation
augmentor = Augmentor((48, 112, 256))

with tf.Session() as sess:
    # Initialise
    sess.run(iterator.initializer)
    # sess.run(iterator_te.initializer)

    batch_images, batch_labels, batch_levels = sess.run([features, labels, levels])
    # print(batch_images.shape)
    # print(sorted(batch_levels[batch_labels==0]))
    # print(sorted(batch_levels[batch_labels==1]))

    aug_batch_images = augmentor.augment_batch(np.copy(batch_images))

    a = 0
    for vol, aug_vol in zip(batch_images, aug_batch_images):
        slice = int(aug_vol.shape[0] / 2)
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
