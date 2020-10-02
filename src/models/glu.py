import tensorflow as tf

class GLU(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(GLU, self).__init__()

        self.dense1 = tf.keras.layers.Dense(d_model)
        self.dense2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        out1 = tf.math.sigmoid(self.dense1(x))
        out2 = self.dense2(x)
        return tf.math.multiply(out1,out2)