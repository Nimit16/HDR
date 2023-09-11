from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import imageio as im
from PIL import Image
import numpy as np
import re
import base64

app = Flask(__name__)

def convertImage(imgData1): 
    imgstr = re.search(b'base64,(.*)',imgData1).group(1) 
    with open('output.png','wb') as output: 
        output.write(base64.b64decode(imgstr))

@app.route('/digit_prediction', methods=['POST'])
def digit_prediction():
    imgData = request.get_data()
    convertImage(imgData)
    x = im.imread('output.png')
    x = np.array(Image.open('output.png').convert('L'))
    x = 255 - x    
    x = np.array(Image.fromarray(x).resize(size=(28, 28)))    
    x = x.reshape(1,28,28,1)    
    x = x.astype('float32')
    x /= 255.0    
    prediction = model.predict(x)
    max_idx = np.argmax(prediction, axis=1)
    max_val = prediction[0][max_idx]
    chart = prediction[0]
    chart = chart.tolist()
    prob = np.around(max_val, 3) * 100
    if prob < 60:
        prediction = 'Huh...'
        probability = 'Please Draw a Better Digit'
    else:
        prediction = " ".join(map(str, max_idx))
        probability = '{}%'.format(" ".join(map(str, prob)))
    return jsonify(result=prediction, probability=probability, chart=chart)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculator')
def calculator():
    return render_template('calculator.html')

if __name__ == '__main__':
    model = load_model('mnist.h5')
    app.run(debug=True)