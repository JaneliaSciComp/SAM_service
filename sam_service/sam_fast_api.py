import os
import base64
import json
import gzip
from io import BytesIO

from typing import Annotated
from fastapi import FastAPI, File, Form, UploadFile, Response, Query
from fastapi.responses import StreamingResponse, RedirectResponse, PlainTextResponse

import torch
from loguru import logger
from sam import SAM

app = FastAPI(
    title="SAM Service",
    license_info={
        "name": "Janelia Open-Source Software License",
        "url": "https://www.janelia.org/open-science/software-licensing",
    },
)

with open('config.json', 'r') as f:
    config = json.load(f)

model_type = config.get('MODEL_TYPE', "vit_h")
module_dir = os.path.dirname(__file__)
checkpoint_file = config.get('CHECKPOINT_FILE', "sam_vit_h_4b8939.pth")
checkpoint = os.path.join(module_dir, checkpoint_file)

device = 'cpu'
if torch.cuda.is_available():
    logger.info('using cuda device for predictions and embedding')
    device = 'cuda'
    # if gpus are specified in the config file, then use a gpu based
    # on the process id, to distribute the load across the gpus by
    # flask process
    gpus = config.get('GPUS', [0])
    gpu_count = len(gpus)
    if gpu_count > 1:
        device = f'cuda:{str(gpus[os.getpid() % gpu_count])}'

sam = SAM(device, model_type, checkpoint)


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
    embedding = sam.buffer_to_embedding(arr_bytes)
    logger.info('embedded image loaded from POST input ...')

    # pass everything to the prediction method
    mask_image = sam.predict_from_embedded(
            embedding,
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
    encoding: str = Query("none", description="compress: Response compressed with gzip")
):
    """accepts an input image and returns a segement_anything box model"""
    logger.info('Started box_model route ...')
    file_data = await image.read()
    img = sam.buffer_to_image(file_data)
    logger.info('image loaded from POST input ...')

    # pass everything to the prediction method
    box_model = sam.get_box_model(img)
    logger.info('model generated ...')

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logger.info('model encoded to base64 string ...')

    if encoding != 'compress':
        return b64_string

    compressed_data = gzip.compress(b64_bytes)
    headers = {
      "Content-Type": "application/gzip",
      "Content-Encoding": "gzip",
    }

    return Response(content=compressed_data, headers=headers)


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
    img = sam.buffer_to_image(file_data)
    mask_image = sam.predict(img, [x, y])
    logger.info('mask returned from predictor ...')
    file_stream = BytesIO(mask_image)
    logger.info('file_stream ready to send ...')
    return StreamingResponse(iter(lambda: file_stream.read(4096), b""), media_type="image/png")
