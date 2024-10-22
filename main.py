import os

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
from layoutlm_preprocess import *
import pytesseract
import json
from collections import defaultdict

app = Flask(__name__)  # instance of flask

CORS(app)

# creating API object
api = Api(app)
app.config['SECRET_KEY'] = 'password'
UPLOAD_FOLDER = '/Users/abhinav/projects/python/kpi-aws/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def iob_to_label(label):
    if label != 'O':
        return label[2:]
    else:
        return ""


label_map = {0: 'B-ANSWER', 1: 'B-HEADER', 2: 'B-QUESTION', 3: 'E-ANSWER', 4: 'E-HEADER', 5: 'E-QUESTION',
             6: 'I-ANSWER', 7: 'I-HEADER', 8: 'I-QUESTION', 9: 'O', 10: 'S-ANSWER', 11: 'S-HEADER', 12: 'S-QUESTION'}


def predict(filename):
    print('model run')
    # #load the model
    model = model_load('layoutlm.pt', 13)
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1_1/bin/tesseract'
    image, words, boxes, actual_boxes = preprocess("uploads/" + filename)
    word_level_predictions, final_boxes, actual_words = convert_to_features(image, words, boxes, actual_boxes, model)

    key_value_pairs = extract_key_value_pairs(word_level_predictions, final_boxes, actual_words)
    return key_value_pairs


@app.route('/extract_key_info', methods=['GET', 'POST'])
# upload image
def upload_image():
    print("file uploaded")
    if 'image' not in request.files:
        return "ERROR:: Please upload a file for processing", 400
    file = request.files['image']
    if file.filename == '':
        return "ERROR:: Empty file uploaded. Please upload a valid image", 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    key_value_pairs = predict(filename)
    return jsonify(key_value_pairs)


def extract_key_value_pairs(word_level_predictions, final_boxes, actual_words):
    # Storage for extracted key-value pairs
    key_value_pairs = defaultdict(str)

    current_key = ""
    current_value = ""
    for prediction, box, word in zip(word_level_predictions, final_boxes, actual_words):
        predicted_label = iob_to_label(label_map[prediction]).lower()

        # print(f"Word: {word}, Prediction: {predicted_label}")

        # If the predicted label is 'question', append to the current key
        if predicted_label == 'question':
            # If there was an active value and a new question starts, finalize the key-value pair
            if current_key and current_value:
                key_value_pairs[current_key.strip()] = current_value.strip()
                current_key = ""  # Reset key
                current_value = ""  # Reset value

            current_key += f" {word}"  # Append word to key

        # If the predicted label is 'answer', append to the current value
        elif predicted_label == 'answer':
            current_value += f" {word}"  # Append word to value

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

    return key_value_pairs


def save_as_json(key_value_pairs):
    key_value_dict = {key: value for key, value in key_value_pairs.items()}

    # Save the dictionary to a JSON file
    with open('output.json', 'w') as json_file:
        json.dump(key_value_dict, json_file, indent=4)

    print("JSON file created successfully!")

    return


if __name__ == '__main__':
    app.run(debug=True)
