from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flasgger import Swagger
from flask import Flask, request, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from wtforms import IntegerField, StringField, SubmitField, SelectField, DecimalField, FileField
from wtforms.validators import Required
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import tensorflow as tf

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/#uploading-files     <---- Finish going thru tutorial
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
swagger = Swagger(app)


# Initialize Form Class
class ModelSelection(FlaskForm):
    photo = FileField("""Upload image to predict:""", validators=[FileRequired()])
    dropdown_list = [('1', 'Predict Digit'), ('2', 'Flower Reccognition'), ('3', 'Dog vs. Cat')]
    selection = SelectField('Select Model', choices=dropdown_list, default=1)
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = ModelSelection()
    if request.method == 'POST':
        # if form.validate_on_submit():  # activates this if/when I hit submit!
        #     print('A valid form was submitted!')
        #     if form.selection.data == 'Predict Digit':
        #         print('Predict Digit')
        #         img = form.photo.data
        #         prediction = predict_digit(photo=img)
        #         return render_template('model.html', form=form, prediction=prediction)

        # else:
        #     print('Something is not right {}'.format(form.errors))
        #     return render_template('model.html', form=form, prediction=-1)
        if 'image' in request.files:
            print('Predict Digit')
            img = request.files['image']
            prediction = predict_digit(photo=img)
            return render_template('model.html', form=form, prediction=prediction)
    else:
        return render_template('model.html', form=form, prediction=-1)


def predict_digit(photo):
    # Important to use same graph used by previous model
    global model
    model = load_model('./model.h5')
    global graph
    graph = tf.get_default_graph()

    # image.load_img allows to rescale image on load.
    img = image.load_img(photo, target_size=(28, 28), color_mode = 'grayscale')
    x = image.img_to_array(img).reshape((1, 28, 28))
    # Need to use this function because model.predict expects a batch of samples instead of just 1
    x = np.expand_dims(x, axis=0)

    with graph.as_default():
        prediction = str(np.argmax(model.predict(x)[0]))

    return prediction


@app.route('/predict_digit', methods=['POST'])
def predict_digit_api():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    form = ImageUploadForm(csrf_enabled=False)
    if form.validate_on_submit():
        if len(request.FILES) > 0:
            digit_image = request.FILES['image']

            # Important to use same graph used by previous model
            global model
            model = load_model('./model.h5')
            global graph
            graph = tf.get_default_graph()

            # image.load_img allows to rescale image on load.
            img = image.load_img(digit_image, target_size=(28, 28), color_mode = 'grayscale')
            x = image.img_to_array(img).reshape((1, 28, 28))
            # Need to use this function because model.predict expects a batch of samples instead of just 1
            x = np.expand_dims(x, axis=0)

            with graph.as_default():
                session['prediction'] = str(np.argmax(model.predict(x)[0]))
            return redirect(url_for('model'))

    return render_template('model.html', form=form, **session)


app.secret_key = 'super_secret_key'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
    #app.run()
