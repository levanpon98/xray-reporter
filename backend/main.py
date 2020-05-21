import io
import numpy as np
import flask
import json
import utils
import models
import config
import pickle
import tensorflow as tf
from flask import request
from PIL import Image
from flask_cors import CORS, cross_origin
from flask_cors import CORS

app = flask.Flask("__main__")
CORS(app)
cors = CORS(app, resources={
    r'/*': {
        'origins': '*'
    }
})

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
encoder = models.CNN_Encoder(config.embedding_dim)
decoder = models.RNN_Decoder(config.embedding_dim, config.units, config.vocab_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = "./saved/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
ckpt.restore(ckpt_manager.latest_checkpoint)

image_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


@app.route("/")
def home():
    return flask.render_template("index.html", token="Hello world")


@app.route('/predict', methods=["POST"])
def predict():
    data = {'success': False}
    print(request.files.getlist("images[]"))
    if request.files.getlist("images[]"):
        images = request.files.getlist("images[]")
        items = []
        for image in images:
            item = {}
            image = image.read()
            image = Image.open(io.BytesIO(image))
            image = utils.preprocess_image(image)
            item['image'] = utils.encode_image(image)
            image_preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
            out = utils.evaluate(image_preprocess, image_features_extract_model, decoder, encoder, tokenizer,
                                 config.max_length)
            item['predict'] = ' '.join(out)

            items.append(item)

        data['success'] = True
        data['data'] = items

    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)


app.run(debug=True)
