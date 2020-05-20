import os
import io
import json
import config
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import img_to_array


class NumpyEncoder(json.JSONEncoder):
    '''
    Encoding numpy into json
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_model():
    base_model = InceptionResNetV2(weights=None, include_top=False,
                                   input_tensor=tf.keras.layers.Input(shape=(config.image_size, config.image_size, 3)))

    head_model = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(head_model)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(config.num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(os.path.join(config.model_path, config.model_name))
    return model


def preprocess_image(image):
    image = image.resize((config.image_size, config.image_size))
    image = img_to_array(image)
    if image.shape[2] == 1:
        image = np.dstack([image] * 3)
    else:
        image = image[:, :, :3]

    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    return image


def encode_image(image):
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img


def evaluate(image, extract_model, decoder, encoder, tokenizer, max_length, ):
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(preprocess_image(image), 0)
    img_tensor_val = extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result
