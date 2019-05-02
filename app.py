from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from VQA_Attack.demo import main
from flask_wtf import FlaskForm
from wtforms import StringField
import json


url = []
file_path = []
aux_path = []

import os

class Question(FlaskForm):
    question = StringField('Question')


app = Flask(__name__)
dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'vqademo'

app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  


@app.route('/', methods=['GET', 'POST'])
def index():
    
    if "file_urls" not in session:
        session['file_urls'] = []
    file_urls = session['file_urls']

    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            filename = photos.save(
                file,
                name=file.filename    
            )
            
            file_path.append(filename)
            aux_path.append(filename)
            file_urls.append(photos.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    return render_template('index.html')


@app.route('/results', methods=['POST', 'GET'])
def results():
    
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    file_urls = session['file_urls']
    url.append(session['file_urls'])
    print('file_urls'+ str(file_urls))
    print('url'+ str(url))
    session.pop('file_urls', None)
    aux_path.pop()
    return render_template('results.html', file_urls=file_urls)


@app.route('/prediction', methods=['POST'])
def question():
   if request.method == 'POST':
        question = request.form.get('question')
        print(question)
        if question is not None:
            urls = url.pop()
            print (urls)
            print (file_path)
            print (os.getcwd()+'/uploads/'+file_path[0])
            fileloc = (os.getcwd()+'/uploads/'+file_path[0])
            output=main(fileloc, question)
            file_path.pop()
            print("out")
            print(output['answer_prob'])
            return render_template('prediction.html', file_urls=urls,answer=zip(output['answer_prob'],output['answer']))

