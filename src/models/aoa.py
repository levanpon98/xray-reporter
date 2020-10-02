import tensorflow as tf
from models.glu import GLU

class AoaLayer(tf.keras.layers.Layer):
    def __init__(self, d_model,dropout_aoa = 0.3):
        super(AoaLayer, self).__init__()

        self.dropout_aoa = tf.keras.layers.Dropout(dropout_aoa)
        self.glu = GLU(d_model)
        self.refine = tf.keras.layers.Dense(d_model)
    def call(self, q, x):
        x = self.dropout_aoa(tf.concat([x,q], axis =-1))
        x = self.glu(self.refine(x))
        return x