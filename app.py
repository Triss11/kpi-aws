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
app.config['UPLOAD_PHOTOS_DEST'] = "C:\\Users\\sohin\\MLAPI\\uploads"

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

@app.route('/upload/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_PHOTOS_DEST'], filename)

#upload image
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)

#prediction api cell
class prediction(Resource):
    def get(self, doc):
        #doc = request.args.get('doc')
        doc = doc.convert("RGB")
        print(doc)
        #load the model
        model = model_load('layoutlm.pt',13)
        image, words, boxes, actual_boxes = preprocess(doc)
        word_level_predictions, final_boxes, actual_words = convert_to_features(image, words, boxes, actual_boxes,
                                                                                model)
        return word_level_predictions, final_boxes, actual_words

#data api
class getData(Resource):
    def get(self):
        df = pd.read_excel('data.xlsx')
        res = df.to_json(orient = 'records')
        return res

api.add_resource(getData, '/api')
api.add_resource(prediction, '/prediction/<doc>')
if __name__ == '__main__':
    app.run(debug=True)