from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('glaucoma_model.keras')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Convert prediction to class label and percentage
    if prediction >= 0.5:
        result = "Glaucoma Affected"
        percentage = round(prediction[0][0] * 100, 2)
        message = 'You Should Consult With A Doctor'
    else:
        result = "Normal"
        percentage = 100.0
        message = 'You Are Perfectly Fine'
    return render_template('result.html', result=result, percentage=percentage, message=message)

if __name__ == '__main__':
    app.run(debug=True)