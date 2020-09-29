import tensorflow as tf

from src import config


class Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                             input_shape=(config.image_height, config.image_height,
                                                                          config.image_channels))
        inputs = base_model.input
        outputs = base_model.layers[-1].output
        self.extract = tf.keras.Model(inputs, outputs)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x, training=None):
        self.extract.summary()
        exit()
        x = self.extract(x)
        print(x)
        exit()
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.fc(x, training=training)
        x = tf.nn.relu(x)
        return x
