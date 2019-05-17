import tensorflow as tf

class Pipeline:
    def __init__(self, decode, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.decode = decode

    def create(self, volume_shape=(5, 256, 256), batch_size=10, test_size=10):
        # Train pipeline
        dataset = tf.data.TFRecordDataset(self.train_data).map(self.decode)
        dataset = dataset.repeat(None)
        dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        # Dataset iterator
        iterator = dataset.make_initializable_iterator()

        # Test pipeline
        dataset_te = tf.data.TFRecordDataset(self.test_data).map(self.decode)
        dataset_te = dataset_te.batch(test_size)
        iterator_te = dataset_te.make_initializable_iterator()

        return iterator, iterator_te
