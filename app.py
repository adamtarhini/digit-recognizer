from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = keras.models.load_model('mnist_model.h5')  # ← add this!

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)  # ← add this!from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = keras.models.load_model('mnist_model.h5')  # ← add this!

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)  # ← add this!