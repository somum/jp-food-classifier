import cv2
import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)
CORS(app)
CATEGORIES = ['ラメン','すし','天ぷら']

json_file = open("structure_food_classifier.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weight_food_classifier.h5")

@app.route('/food-classifier', methods=['GET', 'POST'])
@cross_origin()
def food_classifier():
  if request.method == 'POST':
        if request.files.get('file'):
            img_data =request.files['file']
            img_data = Image.open(img_data)
            img_data = np.array(img_data)
            img_data = cv2.resize(img_data,(128,128))
            img_data = np.expand_dims(img_data, axis = 0)
            prediction = model.predict(img_data)
            pred_class = CATEGORIES[np.argmax(prediction)]

            pred_result={'item_name':pred_class}
  
  return jsonify(pred_result)


@app.route('/', methods=['GET'])
@cross_origin()
def index():
  if request.method == 'GET':
    Message={'message':'Welcome to Japanese Food Classifier API'}
  return jsonify(Message)



if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
