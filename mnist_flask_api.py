from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flasgger import Swagger
from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)
swagger = Swagger(app)



@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    # Important to use same graph used by previous model
    global model
    model = load_model('./model.h5')
    global graph
    graph = tf.compat.v1.get_default_graph() 

    # image.load_img allows to rescale image on load.
    img = image.load_img(request.files['image'], target_size=(28, 28), color_mode = 'grayscale')
    x = image.img_to_array(img).reshape((1,28,28))
    # Need to use this function because model.prdedict expects a batch of samples instead of just 1
    x = np.expand_dims(x, axis=0)

    with graph.as_default():
        pred = model.predict(x)[0]
    return str(np.argmax(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
    #app.run()
