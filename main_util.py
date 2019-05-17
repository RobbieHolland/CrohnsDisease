from dltk.io.preprocessing import *

import tensorflow as tf
import numpy as np

def generate_decode_function(feature_shape, feature_name):
    def decode_record(serialized_example):
        feature_key = f'train/{feature_name}'
        features = tf.parse_single_example(
            serialized_example,
            features={feature_key: tf.FixedLenFeature(feature_shape, tf.float32),
                      'train/label': tf.FixedLenFeature([], tf.int64),
                      'data/index': tf.FixedLenFeature([], tf.int64)})

        return features[feature_key], features['train/label']#, features['data/index']
    return decode_record

def parse_labels(labels):
    return [[0, 1] if level > 0 else [1, 0] for level in labels]

def parse_test_features(features, feature_shape):
    parsed_features = []
    for i, feature in enumerate(features):
        diff = (np.array(feature.shape) - np.array(feature_shape)).astype(int)
        a = [int(round(d / 2)) for d in diff]
        parsed_features.append(whitening(feature[a[0]:-(diff[0]-a[0]), a[1]:-(diff[1]-a[1]), a[2]:-(diff[2]-a[2])]))
    return np.array(parsed_features)

def prediction_class_balance(preds):
    return np.sum(preds) / len(preds)

def accuracy(labels, preds):
    arg_labels = np.argmax(labels, axis=1)
    l_string = ''.join(str(x) for x in arg_labels)
    p_string = ''.join(str(x) for x in preds)
    print(f'Label:      {l_string}')
    print(f'Prediction: {p_string}')
    return np.sum(np.array(arg_labels) == np.array(preds)) / len(arg_labels)

def print_statistics(loss, accuracy, prediction_balance):
    print('Loss:               ', loss)
    print('Accuracy:           ', accuracy)
    print('Prediction balance: ', prediction_balance)

# Test
def test_accuracy(sess, network, batch, iterator_te, iterator_te_next, feature_shape):
    accuracies, all_preds, losses = [], [], []
    summary_te = None

    # Iterate over whole test set
    print()
    print('Test statistics')
    while (True):
        try:
            batch_images, batch_labels = sess.run(iterator_te_next)
            parsed_batch_labels, parsed_batch_features = parse_labels(batch_labels), parse_test_features(batch_images, feature_shape)

            loss_te, summary_te, preds = sess.run([network.summary_loss, network.summary, network.predictions],
                                            feed_dict={network.batch_features: parsed_batch_features,
                                                       network.batch_labels: parsed_batch_labels})
            accuracies.append(accuracy(parsed_batch_labels, preds))
            losses.append(loss_te)
            all_preds += preds.tolist()

        except tf.errors.OutOfRangeError:

            sess.run(iterator_te.initializer)
            average_accuracy = np.average(accuracies)
            print_statistics(np.average(losses), average_accuracy, np.average(prediction_class_balance(all_preds)))
            print()

            return summary_te, average_accuracy, all_preds
