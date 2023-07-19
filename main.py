from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

#Tensorflow and Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
#Image Processing  libraries
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'image_model2.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

# writing the ela function
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image
#end of ela code

#Another function prepare image
def prepare_image(image_path):
    image_size= (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0
#end of prepare image

def model_predict(img_path, model):
    # Preprocessing the image
    img = prepare_image(img_path)
    image = img.reshape(-1, 128, 128, 3)
    x = image
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'source', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        
        #Now predicting the new images by loading the saved model
        class_names = ['Tampered', 'Real']
        y_pred_class = np.argmax(preds, axis = 1)[0]
        result = str(f'Predicted As {class_names[y_pred_class]}  {np.amax(preds) * 100:0.2f} %')
        print(y_pred_class)
        print(result)
        return result
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        ##pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        ##result = str(pred_class[0][0][1])               # Convert to string
        
    return None


if __name__ == '__main__':
    app.run(debug=True)

