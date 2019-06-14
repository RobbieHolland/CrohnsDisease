from sklearn.metrics import classification_report
from dltk.io.preprocessing import *
from hard_soft_map import generate_batch_maps

import tensorflow as tf
import numpy as np

def generate_decode_function(feature_shape, feature_name, localisation):
    def decode_record(serialized_example):
        feature_key = f'train/{feature_name}'
        schema = {feature_key: tf.FixedLenFeature(feature_shape, tf.float32),
                  'train/label': tf.FixedLenFeature([], tf.int64),
                  'data/index': tf.FixedLenFeature([], tf.int64)}

        if localisation:
            schema['train/ileum_coords'] = tf.FixedLenFeature([3], tf.float32)
        features = tf.parse_single_example(
            serialized_example,
            features=schema)

        record = [features[feature_key], features['train/label']]
        if localisation:
            record.append(features['train/ileum_coords'])
        else:
            record.append(np.array([0, 0, 0]))
        return record

    return decode_record

def binarise_labels(labels):
    return [int(label > 0) for label in labels]

def parse_labels(labels):
    return [[0, 1] if level > 0 else [1, 0] for level in labels]

def prediction_class_balance(preds):
    return np.sum(preds) / len(preds)

def accuracy(true_labels, binary_preds):
    tl_string = ''.join(str(x) for x in true_labels)
    binary_labels = binarise_labels(true_labels)
    bl_string = ''.join(str(x) for x in binary_labels)
    p_string = ''.join(str(x) for x in binary_preds)
    print(f'True Label:        {tl_string}')
    print(f'Binary Label:      {bl_string}')
    print(f'Binary Prediction: {p_string}')
    return np.sum(np.array(binary_labels) == np.array(binary_preds)) / len(binary_labels)

def report(labels, preds):
    if len(set(preds)) > 1:
        return classification_report(labels, preds, target_names=['healthy', 'abnormal'])
    return 'Only one class predicted'

def print_statistics(loss, labels, preds):
    print('Loss:               ', loss)
    print('Prediction balance: ', prediction_class_balance(preds))
    print(report(labels, preds))

# Test
def test_accuracy(sess, network, batch, iterator_te, iterator_te_next, augmentor, feature_shape):
    accuracies, all_labels, all_preds, losses = [], [], [], []
    summary_te = None

    # Iterate over whole test set
    print('Test statistics')
    while (True):
        try:
            batch_images, batch_labels, batch_coords = sess.run(iterator_te_next)
            binary_labels = binarise_labels(batch_labels)
            parsed_batch_features, batch_coords = augmentor.parse_test_features(batch_images, batch_coords)
            hard_soft_maps = generate_batch_maps(batch_coords, np.array(feature_shape) // 2)

            loss, summary_te, preds = sess.run([network.summary_loss, network.summary, network.predictions],
                                            feed_dict={network.batch_features: parsed_batch_features,
                                                       network.batch_labels: parse_labels(binary_labels),
                                                       network.batch_coords: batch_coords,
                                                       network.batch_hard_soft_maps: hard_soft_maps})
            losses += [loss] * len(batch_labels)
            all_preds += preds.tolist()
            all_labels += batch_labels.tolist()
            print(batch_coords[0], preds[0])

        except tf.errors.OutOfRangeError:
            sess.run(iterator_te.initializer)
            overall_loss = np.average(losses)

            return summary_te, overall_loss, all_preds, binarise_labels(all_labels)
