from flask import Flask, request, jsonify
import tensorflowjs as tfjs
import numpy as np
from PIL import Image
"""
import tensorflow as tf

model = tf.keras.models.load_model('CNNmodel.h5')
tfjs.converters.save_keras_model(model, 'tfjs_model')
"""
app = Flask(__name__)
model = tfjs.converters.load_keras_model('tfjs_model/model.json')

@app.route('/predict', methods=['POST','GET'])

def predict():
    # Read the image from the request
    image = Image.open(request.files['image'])

    # Resize and normalize the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    # Make the prediction
    #prediction = model.predict(np.array([image]))

    # Return the prediction as JSON prediction.tolist()
    #width,height=image.size
    return jsonify({'prediction': len(image)})
app.run()