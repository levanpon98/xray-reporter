import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3

import config


class Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        self.extract = EfficientNetB3(weights='imagenet', drop_connect_rate=0.4, include_top=False,
                                      input_shape=(config.image_height, config.image_width, config.image_channels))

        shape = self.extract.layers[-1].output_shape
        self.reshape = tf.keras.layers.Reshape(target_shape=(shape[1] * shape[2], shape[3]))
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.extract(x)
        x = self.reshape(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
