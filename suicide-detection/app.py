# app.py
from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('suicide_detection_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    result = 'Suicide' if prediction > 0.5 else 'Non-suicide'
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)

# templates/index.html remains the same as in the previous artifact