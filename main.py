from flask import Flask, render_template, redirect, request
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50
from keras.models import load_model
from keras.utils import load_img, img_to_array 
import numpy as np
import os

from werkzeug.utils import secure_filename



app = Flask(__name__)

MODEL_PATH = 'models/model_resnet.h5'
model = load_model(MODEL_PATH)

#model = ResNet50(weights='imagenet') #For the first time
#model.save('models/model_resnet.h5')

def model_predict(img_path, model):
    img  = load_img(img_path, target_size=(224,224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/demo")
def demo(methods = ['GET']):
    return render_template('demo.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
       
        file_path = os.path.join('preds', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None

if __name__ == "__main__":
    app.run(debug=True)




