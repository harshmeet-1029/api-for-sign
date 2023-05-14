import os
import io
import json
import base64
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from flask import Flask, request
# Initialize Flask app
app = Flask(__name__)
# initialize mediapipe
mpHands = mp.solutions.hands

# initialize Hands from mediapipe

# config
maxNumHands = 1
minDetectionConfidence=0.7

hands = mpHands.Hands(
    max_num_hands=maxNumHands,
    min_detection_confidence=minDetectionConfidence
    )

# initialize drawing_utils from mediapipe 
mpDraw = mp.solutions.drawing_utils
# Load TFLite model
# Load the gesture recognizer model
model = load_model('Modal')
# Load class names
f = open('model.txt', 'r')
# converting list into array of strings
classNames = f.read().split('\n')
f.close()
# Define API endpoint
@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get image from request data
    image_data = request.get_data()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = np.array(image)
    result = hands.process(image)
    x, y, c = image.shape
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
            className = ''
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            temp =''
            
            if className!="":
                print(className) 
                result={'classname':className}
            else:
                result={'classname':className}
                         

    return json.dumps(result)

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
