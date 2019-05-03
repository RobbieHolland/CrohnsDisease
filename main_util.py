import tensorflow as tf
import numpy as np

def generate_decode_function(feature_shape, feature_name):
    def decode_record(serialized_example):
        feature_key = f'train/{feature_name}'
        features = tf.parse_single_example(
            serialized_example,
            features={feature_key: tf.FixedLenFeature(feature_shape, tf.float32),
                      'train/label': tf.FixedLenFeature([], tf.int64)})
        return features[feature_key], features['train/label']
    return decode_record

def parse_labels(labels):
    return [[0, 1] if level > 0 else [1, 0] for level in labels]

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
def test_accuracy(sess, network, batch, iterator_te, iterator_te_next, augmentor):
    accuracies = []
    prediction_balances = []
    losses = []
    summary_te = None

    # Iterate over whole test set
    print()
    print('Test statistics')
    while (True):
        try:
            batch_images, batch_labels = sess.run(iterator_te_next)
            parsed_batch_labels = parse_labels(batch_labels)

            loss_te, summary_te, preds = sess.run([network.summary_loss, network.summary, network.predictions],
                                            feed_dict={network.batch_features: batch_images,
                                                       network.batch_labels: parsed_batch_labels})
            accuracies.append(accuracy(parsed_batch_labels, preds))
            losses.append(loss_te)
            prediction_balances.append(prediction_class_balance(preds))

        except tf.errors.OutOfRangeError:

            sess.run(iterator_te.initializer)
            average_accuracy = np.average(accuracies)
            print_statistics(np.average(losses), average_accuracy, np.average(prediction_balances))
            print()

            return summary_te, average_accuracy
