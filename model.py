import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape


class VAE(tf.keras.Model):
    def __init__(self, net_type: str = 'RNN'):
        super(VAE, self).__init__()
        assert net_type in ['DNN', 'RNN', 'CNN']
        if net_type == "DNN":
            InputLayer(input_shape=[20*52]),
        if net_type == "RNN":

        if net_type == "CNN"
