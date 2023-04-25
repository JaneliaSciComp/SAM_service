from io import BytesIO
from flask import Flask, render_template, request, redirect, flash, send_file
import cv2
import numpy as np

import paintera_sam.sam_model as model

app = Flask(__name__)

app.secret_key = "cdas69786cdsa^%#FDS^%$gfd65#$dfs#@$#@?><i:OI)(*-"

sam_predictor = model.new_predictor(device='cpu')


def predict(cv_image, coords):
    print("running prediction")
    sam_predictor.set_image(cv_image, image_format='BGR')
    masks = model.predict_current_image(sam_predictor, *coords, cv_image, show=False)
    result, buffer = cv2.imencode('.png', masks[0] * 255)
    return buffer


@app.route("/")
def service_docs():
    """render an input form and documentation"""
    return render_template("main.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    """accepts an input image and coordinates and returns a prediction"""
    # get the image from the request object
    # check if the post request has the file part
    if "image" not in request.files:
        flash("No file part")
        return redirect("/")

    file = request.files["image"]
    file_data = file.read()

    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

    # pass everything to the prediction method
    mask_image = predict(img, [x_coordinate, y_coordinate])
    print(mask_image)

    # return the mask.
    # file_stream = BytesIO(file_data)
    file_stream = BytesIO(mask_image)
    return send_file(file_stream, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
