import io
import numpy as np
import flask
import json
import utils
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

model = utils.get_model()


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
            image_norm = np.expand_dims(image, axis=0).astype('float') / 255.

            out = model.predict(image_norm)
            item['predict'] = [{'label': 'Covid-19', 'probability': round(out[0][0], 3)},{'label': 'Normal', 'probability': round(out[0][1], 3)}]
            item['image'] = utils.encode_image(image)
            items.append(item)

        data['success'] = True
        data['data'] = items

    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)


app.run(debug=True)

