import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3

import config


class Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        self.extract = EfficientNetB3(weights='imagenet', include_top=False, drop_connect_rate=0.4)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x, training=None):
        x = self.extract(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.fc(x, training=training)
        x = tf.nn.relu(x)
        return x

