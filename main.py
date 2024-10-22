from flask import Flask,request,render_template, send_from_directory, url_for
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
from layoutlm_preprocess import *
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_wtf import FlaskForm
import pytesseract
import json
from collections import defaultdict

app = Flask(__name__) #instance of flask

CORS(app)

#creating API object
api = Api(app)
app.config['SECRET_KEY'] = 'password'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos',IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    print("file got")
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return ""

label_map= {0: 'B-ANSWER', 1: 'B-HEADER', 2: 'B-QUESTION', 3: 'E-ANSWER', 4: 'E-HEADER', 5: 'E-QUESTION', 6: 'I-ANSWER', 7: 'I-HEADER', 8: 'I-QUESTION', 9: 'O', 10: 'S-ANSWER', 11: 'S-HEADER', 12: 'S-QUESTION'}

#@app.route('/predict-image/<filename>', methods=['GET'])
def predict(filename):
    print('model run')
    # #load the model
    model = model_load('layoutlm.pt', 13)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\Tesseract.exe'
    image, words, boxes, actual_boxes = preprocess("uploads/"+filename)
    word_level_predictions, final_boxes, actual_words = convert_to_features(image, words, boxes, actual_boxes,model)
    
    extract_key_value_pairs(word_level_predictions,final_boxes,actual_words)



@app.route('/', methods=['GET','POST'])
#upload image
def upload_image():
    print("file uploaded")
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        predict(filename)
    else:
        file_url = None

    return render_template('index.html', form=form, file_url=file_url)

def extract_key_value_pairs(word_level_predictions,final_boxes,actual_words):
    # Storage for extracted key-value pairs
    key_value_pairs = defaultdict(str)
    
    current_key = ""
    current_value = ""
    is_key_active = False  # Tracks if we are appending to a key or value
    
    for prediction, box, word in zip(word_level_predictions, final_boxes, actual_words):
        predicted_label = iob_to_label(label_map[prediction]).lower()
    
        #print(f"Word: {word}, Prediction: {predicted_label}")
    
        # If the predicted label is 'question', append to the current key
        if predicted_label == 'question':
            # If there was an active value and a new question starts, finalize the key-value pair
            if current_key and current_value:
                key_value_pairs[current_key.strip()] = current_value.strip()
                current_key = ""  # Reset key
                current_value = ""  # Reset value
    
            current_key += f" {word}"  # Append word to key
            is_key_active = True  # Indicate that we're building the key
    
        # If the predicted label is 'answer', append to the current value
        elif predicted_label == 'answer':
            current_value += f" {word}"  # Append word to value
            is_key_active = False  # Indicate that we're building the value
    
        # If a non-question/answer label appears, finalize the current key-value pair
        else:
            if current_key and current_value:
                key_value_pairs[current_key.strip()] = current_value.strip()
            current_key = ""  # Reset for the next key
            current_value = ""  # Reset for the next value
    
    # If any key-value pair remains unprocessed, add it
    if current_key and current_value:
        key_value_pairs[current_key.strip()] = current_value.strip()
    
    # Print the extracted key-value pairs
    print("\nExtracted Key-Value Pairs:")
    for key, value in key_value_pairs.items():
        print(f"{key.capitalize()}: {value}")

    save_as_json(key_value_pairs)

def save_as_json(key_value_pairs):
    key_value_dict = {key: value for key, value in key_value_pairs.items()}

    # Save the dictionary to a JSON file
    with open('output.json', 'w') as json_file:
        json.dump(key_value_dict, json_file, indent=4)
    
    print("JSON file created successfully!")

if __name__ == '__main__':
    app.run(debug=True)
