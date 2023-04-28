import logging
import base64
from io import BytesIO
import os
from flask import Flask, render_template, request, redirect, flash, send_file, jsonify
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

import paintera_sam.sam_model as model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

app = Flask(__name__)
app.secret_key = "cdas69786cdsa^%#FDS^%$gfd65#$dfs#@$#@?><i:OI)(*-"


logging.info('Creating new predictor...')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam_predictor = model.new_predictor(device=device)

logging.info('Created new predictor')

def predict(cv_image, coords):
    logging.info("running prediction")
    sam_predictor.set_image(cv_image, image_format='BGR')
    masks = model.predict_current_image(sam_predictor, *coords, cv_image, show=False)
    _, buffer = cv2.imencode('.png', masks[0] * 255)
    return buffer

def get_box_model(cv_image):
    logging.info("embedding image")
    sam_predictor.set_image(cv_image, image_format='BGR')
    return sam_predictor.get_image_embedding().cpu().numpy()


@app.route("/")
def service_docs():
    """render an input form and documentation"""
    return render_template("main.html")

@app.route("/box_model", methods=["POST"])
def box_model():
    """accepts an input image and returns a segement_anything box model"""
    logging.info('Started prediction route ...')
    # get the image from the request object
    # check if the post request has the file part
    if "image" not in request.files:
        flash("No file part")
        return redirect("/")

    file = request.files["image"]
    file_data = file.read()

    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logging.info('image loaded from POST input ...')

    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return redirect("/")

    # pass everything to the prediction method
    box_model = get_box_model(img)
    logging.info('model generated ...')

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logging.info('model encoded to base64 string ...')
    response = jsonify([b64_string])
    response.headers['Content-Type'] = 'application/json'
    return response



@app.route("/prediction", methods=["POST"])
def prediction():
    """accepts an input image and coordinates and returns a prediction"""
    logging.info('Started prediction route ...')
    # get the image from the request object
    # check if the post request has the file part
    if "image" not in request.files:
        flash("No file part")
        return redirect("/")

    file = request.files["image"]
    file_data = file.read()

    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logging.info('image loaded from POST input ...')

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return redirect("/")

    # get the point coordinates
    if "x" not in request.form:
        flash("No x coordinate specified")
        return redirect("/")
    x_coordinate = request.form.get("x")
    if "y" not in request.form:
        flash("No y coordinate specified")
        return redirect("/")
    y_coordinate = request.form.get("y")
    logging.info('coordinates loaded from POST input ...')

    # pass everything to the prediction method
    mask_image = predict(img, [x_coordinate, y_coordinate])
    logging.info('mask returned from predictor ...')

    # return the mask.
    file_stream = BytesIO(mask_image)
    logging.info('file_stream ready to send ...')
    return send_file(file_stream, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='5050')
