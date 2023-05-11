import os
import base64
import warnings
import json
from io import BytesIO
import logging
from fastapi import FastAPI, File, Form, UploadFile, Response
from typing import Annotated
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, PlainTextResponse
import torch

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import paintera_sam.sam_model as model
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from loguru import logger

app = FastAPI(
    title="SAM Service",
    license_info={
        "name": "Janelia Open-Source Software License",
        "url": "https://www.janelia.org/open-science/software-licensing",
    },
)


with open('config.json', 'r') as f:
    config = json.load(f)

logger.info('Creating new predictor...')
module_dir = os.path.dirname(__file__)

model_type = "vit_h"
checkpoint = os.path.join(module_dir, "sam_vit_h_4b8939.pth")
device = 'cpu'
if torch.cuda.is_available():
    logger.info('using cuda device for predictions and embedding')
    device = 'cuda'
    # if gpus are specified in the config file, then use a gpu based
    # on the process id, to distribute the load across the gpus by
    # flask process
    gpu_count = config.get('GPU_COUNT', 1)
    if gpu_count > 1:
        device = f'cuda:{str(os.getpid() % gpu_count)}'


sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)
logger.info('Created new predictor')

def create_onnx_runtime():
    logger.info('Creating new onnx runtime...')

    # there needs to be a seperate onnx_sam model that is sent to the cpu,
    # because the onnx export occurs on the cpu and not the gpu.
    onnx_sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_sam.to(device='cpu')

    onnx_model_path = f"{str(os.getpid())}.sam_onnx.onnx"

    onnx_model = SamOnnxModel(onnx_sam, return_single_mask=True)

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

    onnx_model_quantized_path = f"{str(os.getpid())}.sam_onnx_quantized_example.onnx"

    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
    onnx_model_path = onnx_model_quantized_path
    logger.info('Created new onnx runtime...')

    return onnxruntime.InferenceSession(onnx_model_path)


ort_session = create_onnx_runtime()



def predict(cv_image, coords):
    logger.info(f"running prediction on {device}")
    sam_predictor.set_image(cv_image, image_format='BGR')
    masks = model.predict_current_image(sam_predictor, *coords, cv_image, show=False)
    _, buffer = cv2.imencode('.png', masks[0] * 255)
    return buffer


def get_box_model(cv_image):
    logger.info(f"embedding image on {device}")
    sam_predictor.set_image(cv_image, image_format='BGR')
    return sam_predictor.get_image_embedding().cpu().numpy()


def predict_from_embedded(image_embedding, coords, img_dimensions):
    logger.info(f"running embedded prediction on {device}")
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


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse("/docs")


@app.post("/from_model")
async def from_embedded_model(
    model: Annotated[UploadFile, File()],
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
    img_x: Annotated[int, Form()],
    img_y: Annotated[int, Form()]
):
    """accepts an embedded image model, coordinates and returns a mask"""
    logger.info('Started from model route ...')
    file_data = await model.read()
    arr_bytes = base64.b64decode(file_data)
    nparr = np.frombuffer(arr_bytes, dtype=np.float32)
    # restore the orginal shape of the numpy array, because frombuffer will
    # only create a one dimensional array.
    reshaped = nparr.reshape((1,256,64,64))
    logger.info('embedded image loaded from POST input ...')

    # pass everything to the prediction method
    mask_image = predict_from_embedded(
            reshaped,
            [x, y],
            [img_y,img_x]
    )
    logger.info('mask returned from predictor ...')

    # return the mask.
    file_stream = BytesIO(mask_image)
    logger.info('file_stream ready to send ...')
    return StreamingResponse(iter(lambda: file_stream.read(4096), b""), media_type="image/png")


@app.post("/embedded_model", response_class=PlainTextResponse)
async def embedded_model(
    image: Annotated[UploadFile, File()],
    response: Response
):
    """accepts an input image and returns a segement_anything box model"""
    logger.info('Started box_model route ...')
    file_data = await image.read()
    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logger.info('image loaded from POST input ...')

    # pass everything to the prediction method
    box_model = get_box_model(img)
    logger.info('model generated ...')

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logger.info('model encoded to base64 string ...')
    return b64_string


@app.post("/prediction")
async def predict_form(
    image: Annotated[UploadFile, File()],
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
):
    """accepts an input image and coordinates and returns a prediction mask"""
    logger.info('Started prediction route ...')
    file_data = await image.read()
    logger.info('image loaded from POST input ...')
    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    mask_image = predict(img, [x, y])
    logger.info('mask returned from predictor ...')
    file_stream = BytesIO(mask_image)
    logger.info('file_stream ready to send ...')
    return StreamingResponse(iter(lambda: file_stream.read(4096), b""), media_type="image/png")
