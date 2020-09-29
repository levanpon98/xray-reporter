import tensorflow as tf


def get_extract_model():
    base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    inputs = base_model.input
    outputs = base_model.layers[-1].output

    return tf.keras.Model(inputs, outputs)
