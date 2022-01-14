import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Reshape, GRU


class classifier(tf.keras.Model):
    def __init__(self, net_type: str = 'RNN'):
        super(classifier, self).__init__()
        assert net_type in ['DNN', 'RNN', 'CNN']
        if net_type == "DNN":
            self.model = tf.keras.Sequential([
                InputLayer(input_shape=[20, 52]),
                Reshape(target_shape=[20*52]),
                Dense(1040, activation='relu'),
                Dense(4, activation='softmax')
                ])
        if net_type == "RNN":
            self.model = tf.keras.Sequential([
                InputLayer(input_shape=[20, 52]),
                GRU(40),
                Dense(4, activation='softmax')
                ])
        if net_type == "CNN":
            self.model = tf.keras.Sequential([
                InputLayer(input_shape=[20, 52]),
                Reshape(target_shape=[20, 52, 1]),
                Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                Flatten(),
                Dense(4, activation='softmax')
            ])

