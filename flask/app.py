import flask
import werkzeug

!git clone https://github.com/CRSohan4/Plant-Disease-Detection.git
# Import OpenCV
import cv2
import os
import numpy as np
# Utility
import itertools
import random
from collections import Counter
from glob import iglob

import matplotlib.pyplot as plt


def load_image(filename):
    img = cv2.imread(os.path.join(data_dir, validation_dir, filename))
    img = cv2.resize(img, (224, 224) )
    img = img /255
    
    return img

def predict(image):
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    # classes = [ i.split("/")[-1] for i in folders ]

    return {classes[class_idx]: probabilities[class_idx]}
  
def get_filename():
    path = validation_dir + "/"
    path += random.choice(classes)
  
    filename = path + "/" + random.choice(os.listdir(path))
    return filename 

data_dir = "/content/Plant-Disease-Detection"
train_dir = data_dir + "/train"
validation_dir = data_dir + "/validation"
categories_dir = data_dir + "/categories"

pixels = 224
IMAGE_SIZE = (pixels, pixels)

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

from tensorflow import keras
model = keras.models.load_model(data_dir + '/saved_models/plant_disease_new_model_100_epochs.hdf5')



# print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
# plt.imshow(img)
# plt.figure(idx)    
# plt.show()

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    
    img = load_image(imagefile.filename)
    prediction = predict(img)
    print()
    print(prediction)
    print()
    
    # imagefile.save(filename)
    return "Image Uploaded Successfully"

app.run(host="0.0.0.0", port=5000, debug=True)