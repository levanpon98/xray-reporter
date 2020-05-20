from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow
import time
import matplotlib.pyplot as plt
import numpy as np
loaded_model = load_model("covid.model (1).h5")
image_path="/content/drive/My Drive/VNOnlineHackathon/xray-data-covid/xray-data/covid/1-s2.0-S0929664620300449-gr2_lrg-b.jpg"

img = image.load_img(image_path, target_size=(456, 456))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0).astype('float') / 255.
gap_weights = loaded_model.layers[-1].get_weights()[0]
gap_weights2 = loaded_model.layers[-3].get_weights()[0]
cam_model  = tensorflow.keras.Model(inputs=loaded_model.input,outputs=(loaded_model.layers[-5].output,loaded_model.layers[-3].output,loaded_model.layers[-1].output))
features,mid, results = cam_model.predict(img)

tick = time.time()
# plt.imshow(img)
features_for_one_img = features[0]
height_roomout = img.shape[1]/features_for_one_img.shape[0]
width_roomout  = img.shape[2]/features_for_one_img.shape[1]
print(int(height_roomout),int(width_roomout))

cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=1)
print(cam_features.shape)

pred = np.argmax(results[0])
print(pred)

plt.figure(facecolor='white')
cam_weights = gap_weights[:,pred]
cam_output  = np.dot(cam_features,gap_weights2)
cam_output  = np.dot(cam_output,cam_weights)
print(cam_output.shape)
tock = time.time()
print(tock -tick)

plt.imshow(np.squeeze(img,0), alpha=0.5)
plt.imshow(cam_output, cmap='jet', alpha=0.5)
plt.show()