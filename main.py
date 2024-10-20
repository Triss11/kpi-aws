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

@app.route('/predict-image/<filename>', methods=['GET'])
def predict(filename):
    print('model run')
    # #load the model
    model = model_load('layoutlm.pt', 13)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\Tesseract.exe'
    image, words, boxes, actual_boxes = preprocess("uploads/"+filename)
    word_level_predictions, final_boxes, actual_words = convert_to_features(image, words, boxes, actual_boxes,model)
    key_value_pairs = []
    current_key = None
    current_value = None

    for prediction, word in zip(word_level_predictions, actual_words):
        predicted_label = iob_to_label(label_map[prediction]).lower()

        if predicted_label == "question":  # Assuming 'question' is the label for key
            if current_key is not None:  # Store previous key-value pair
                key_value_pairs.append((current_key, current_value))
            current_key = word  # Start a new key
            current_value = ""
        elif predicted_label == "answer":  # Assuming 'answer' is the label for value
            if current_value is None:
                current_value = word  # Initialize value
            else:
                current_value += " " + word  # Continue value if it's multi-word
        # Handle cases where the label may not fit question-answer pattern

    if current_key is not None:  # Store the last pair if any
        key_value_pairs.append((current_key, current_value))

    return key_value_pairs



@app.route('/', methods=['GET','POST'])
#upload image
def upload_image():
    print("file uploaded")
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)


if __name__ == '__main__':
    app.run(debug=True)