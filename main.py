import os
import time

import numpy as np
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, render_template, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename, redirect
from wtforms import SubmitField


class PhotoForm(FlaskForm):
    photo = FileField(validators=[FileRequired()])
    submit = SubmitField("Upload")


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = "verysecretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

Bootstrap(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload():
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename != 'favicon.ico':
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(filepath)
    form = PhotoForm()
    if form.validate_on_submit():
        file = form.photo.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('show', file=filename))

    return render_template('index.html', form=form)


@app.route('/<file>')
def show(file):
    colors = sample_colors(os.path.join(UPLOAD_FOLDER, file))
    return render_template('index.html', file=file, colors=colors)


def sample_colors(file_path):
    while not os.path.exists(file_path):
        time.sleep(1)

    uploaded_img = Image.open(file_path)
    img_array = np.array(uploaded_img)
    try:
        colors_array = np.reshape(img_array, (img_array.shape[0] * img_array.shape[1], 3))
    except ValueError:
        rgb_im = uploaded_img.convert('RGB')
        rgb_img_array = np.array(rgb_im)
        colors_array = np.reshape(rgb_img_array, (img_array.shape[0] * img_array.shape[1], 3))

    clt = KMeans(n_clusters=10)
    clt.fit(colors_array)

    colors_df = pd.DataFrame(clt.cluster_centers_)

    top10colors = colors_df.groupby(colors_df.columns.tolist(), as_index=False).size() \
        .sort_values(by='size', ascending=False).reset_index(drop=True)
    top10colors = top10colors.astype({0: "int", 1: "int", 2: "int"})

    rgb_list = [(row[0], row[1], row[2]) for index, row in top10colors.iterrows()]

    return rgb_list


if __name__ == "__main__":
    app.run()
