from flask import Flask, render_template, jsonify, request
from binascii import a2b_base64
import random
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import warnings

app = Flask(__name__)

model = load_model('mnist_model.h5', compile=False)


def prepareImg(number):
    img = cv2.imread(f'uploads/image-{number}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(-1, 28, 28, 1)
    img = img/255.0
    print(img.shape)
    return img





def GetPredict(x):
    print(x)
    pred = model.predict(prepareImg(x))
    warnings.simplefilter("ignore")
    index = np.argmax(pred)
    print(index)
    return int(index)


@app.route('/')
def index():
    print('happy')
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def uploadAndPredict():
    image_b64 = request.values['imageBase64']
    count = random.randint(0, 10000)
    # Removing Prefix from Data URI
    data = image_b64.replace("data:image/png;base64,", '')
    binary_data = a2b_base64(data)
    # Get the Images Saved in Upload Folder
    # Getting Prediction from Conv-model
    fd = open(f'uploads/image-{count}.png', 'wb')
    fd.write(binary_data)
    fd.close()
    chrX = GetPredict(count)

    # Sending the JSON response of the Object
    res = {

        "prediction": chrX,
    }
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)
