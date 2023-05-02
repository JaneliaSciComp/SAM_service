import logging
import base64
from io import BytesIO
import os
from flask import Flask, render_template, request, redirect, flash, send_file, Response
import cv2
import numpy as np
import torch
import warnings

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import paintera_sam.sam_model as model
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

app = Flask(__name__)
app.secret_key = "cdas69786cdsa^%#FDS^%$gfd65#$dfs#@$#@?><i:OI)(*-"

logging.info('Creating new predictor...')

module_dir = os.path.dirname(__file__)

model_type = "vit_h"
checkpoint = os.path.join(module_dir, "sam_vit_h_4b8939.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)

logging.info('Created new predictor')


logging.info('Creating new onnx runtime...')


onnx_model_path = "sam_onnx.onnx"

onnx_model = SamOnnxModel(sam, return_single_mask=True)

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"

quantize_dynamic(
    model_input=onnx_model_path,
    model_output=onnx_model_quantized_path,
    optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
onnx_model_path = onnx_model_quantized_path

ort_session = onnxruntime.InferenceSession(onnx_model_path)


logging.info('Created new onnx runtime...')

def predict(cv_image, coords):
    logging.info("running prediction")
    sam_predictor.set_image(cv_image, image_format='BGR')
    masks = model.predict_current_image(sam_predictor, *coords, cv_image, show=False)
    _, buffer = cv2.imencode('.png', masks[0] * 255)
    return buffer

def predict_from_embedded(image_embedding, coords, img_dimensions):
    logging.info("running embedded prediction")
    input_point = np.array([coords])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = sam_predictor.transform.apply_coords(onnx_coord, img_dimensions).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(img_dimensions, dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > sam_predictor.model.mask_threshold

    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks.shape[-2:]
    mask_image = masks.reshape(h, w, 1) * 255
    _, buffer = cv2.imencode('.png', mask_image)
    return buffer

def get_box_model(cv_image):
    logging.info("embedding image")
    sam_predictor.set_image(cv_image, image_format='BGR')
    return sam_predictor.get_image_embedding().cpu().numpy()


@app.route("/")
def service_docs():
    """render an input form and documentation"""
    return render_template("main.html")

@app.route("/from_model", methods=["POST"])
def from_box_model():
    """accepts an embedded image model, coordinates and returns a mask"""
    logging.info('Started from route ...')
    # get the model from the request object
    if "model" not in request.files:
        flash("No file part")
        return redirect("/")

    file = request.files["model"]
    file_data = file.read()
    arr_bytes = base64.b64decode(file_data)


    nparr = np.frombuffer(arr_bytes, dtype=np.float32)
    # restore the orginal shape of the numpy array, because frombuffer will
    # only create a one dimensional array. 
    reshaped = nparr.reshape((1,256,64,64))
    logging.info('embedded image loaded from POST input ...')

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
    mask_image = predict_from_embedded(reshaped, [x_coordinate, y_coordinate], [1200,1800])
    logging.info('mask returned from predictor ...')

    # return the mask.
    file_stream = BytesIO(mask_image)
    logging.info('file_stream ready to send ...')
    return send_file(file_stream, mimetype="image/png")






@app.route("/embedded_model", methods=["POST"])
def embedded_model():
    """accepts an input image and returns a segement_anything box model"""
    logging.info('Started box_model route ...')
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
    logging.info(box_model.shape)

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logging.info('model encoded to base64 string ...')
    response = Response(b64_string, status=200, mimetype='application/octet-stream')
    response.headers.set('Content-Disposition', 'attachment', filename='embedded_image.txt')
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
