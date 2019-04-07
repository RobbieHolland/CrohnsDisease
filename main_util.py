import tensorflow as tf
import numpy as np

def augment_batch(augmentor, features, labels):
    aug_batch_images = augmentor.augment_batch(features)
    aug_batch_labels = [[0, 1] if polyp > 0 else [1, 0] for polyp in labels]
    return aug_batch_images, aug_batch_labels

def prediction_class_balance(preds):
    return np.sum(preds) / len(preds)

def accuracy(labels, preds):
    arg_labels = np.argmax(labels, axis=1)
    return np.sum(np.array(arg_labels) == np.array(preds)) / len(labels)

def print_statistics(loss, accuracy, prediction_balance):
    print('Loss:               ', loss)
    print('Accuracy:           ', accuracy)
    print('Prediction balance: ', prediction_balance)

# Test
def test_accuracy(sess, network, batch, iterator_te, augmentor):
    accuracies = []
    prediction_balances = []
    losses = []
    summary_te = None

    # Iterate over whole test set
    try:
        while (True):
            batch_images, batch_labels = sess.run(iterator_te.get_next())
            aug_batch_images, aug_batch_labels = augment_batch(augmentor, batch_images, batch_labels)

            loss_te, summary_te, preds = sess.run([network.summary_loss, network.summary, network.predictions],
                                            feed_dict={network.batch_features: aug_batch_images,
                                                       network.batch_labels: aug_batch_labels})
            accuracies.append(accuracy(aug_batch_labels, preds))
            losses.append(loss_te)
            prediction_balances.append(prediction_class_balance(preds))

    except tf.errors.OutOfRangeError:
        average_accuracy = np.average(accuracies)

        sess.run(iterator_te.initializer)
        print()
        print('Test statistics')
        print_statistics(np.average(losses), average_accuracy, np.average(prediction_balances))
        print()

        return summary_te, average_accuracy
