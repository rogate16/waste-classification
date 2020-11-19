import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

model = load_model("model/model.h5")

path = "example/"
path = path + os.listdir(path)[0]
img = load_img(path, target_size=(200,200))
img = img_to_array(img)

img = np.expand_dims(img, axis=0)

result = model.predict(img)

if(result[0][0]==0):
    label = "Organic"
else:
    label = "Recycle"

plt.text(10,-20,label, color="blue")
plt.imshow(load_img(path))
plt.show()
