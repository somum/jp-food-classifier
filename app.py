import cv2
import os
import decimal
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

#classes of foods
CATEGORIES = ['ラメン','すし','天ぷら']

#loading model & weight
json_file = open("structure_food_classifier1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weight_food_classifier1.h5")

#classifier page
@app.route('/food-classifier', methods=['POST'])
@cross_origin()
def food_classifier():
  if request.method == 'POST':
  	    # formdata appended as 'file'
        if request.files.get('file'):
            try:
            	#processing for feeding into model 
                img_data =request.files['file']
                img_data = Image.open(img_data)
                img_data = np.array(img_data)
                img_data = img_data/255
                img_data = cv2.resize(img_data,(256,256))
                img_data = np.expand_dims(img_data, axis = 0)
                #feeding into model
                prediction = np.asarray(model.predict(img_data), dtype = decimal.Decimal)
                #if the threshold value is more than 50% result will be shown
                if(prediction.max() > 0.5):
                    pred_class = CATEGORIES[np.argmax(prediction)]
                    pred_result={'item_name':pred_class}
                    return jsonify(pred_result)
                else:
                    error_message = {'message':'Please try again'}
                    return jsonify(error_message)
            except:
               error_message = {'message':'We are facing some problem. Try after sometime !'}
               return jsonify(error_message)
  
  else:
  	#file checking
    error_message = {'message':'Please upload a valid image file !'}
    return jsonify(pred_result)

#index page message
@app.route('/', methods=['GET'])
@cross_origin()
def index():
  if request.method == 'GET':
    Message={'message':'Welcome to Japanese Food Classifier API'}
  return jsonify(Message)



if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
