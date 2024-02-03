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
import utils

from dataclasses import dataclass
from typing import Callable
from threading import Thread
from queue import Queue
import janus
from collections import defaultdict

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
worker_ids = config.get('WORKERS', [0])
gpus = config.get('GPUS', [])
gpu_count = len(gpus)

device = 'cpu'
if torch.cuda.is_available():
    logger.info('using cuda device for predictions and embedding')
    device = 'cuda'
    
@dataclass
class WorkItem:
    work_function: Callable
    result_queue: janus.SyncQueue
    session_id: str
    stale: bool = False

logger.info(f"Creating {len(worker_ids)} workers backed by {gpu_count} GPUs")

def worker_loop(worker_id:int, work_queue:janus.SyncQueue[WorkItem]):
    """ Loops forever and pulls work from the given work queue.
    """
    worker_device = device
    if worker_device=='cuda' and gpu_count > 1:
        worker_device = f'cuda:{str(gpus[worker_id % gpu_count])}'
    logger.info(f"Created worker thread {worker_id} running on {worker_device}")
    sam = SAM(worker_device, model_type, checkpoint)

    while True:
        item = work_queue.get()
        if item.stale:
            logger.warning(f"Ignoring stale work item for session {item.session_id}")
        else:
            logger.info(f"Worker {worker_id} processing item")
            result = item.work_function(sam)
            logger.info(f"Worker {worker_id} processed item")
            item.result_queue.put(result)
        work_queue.task_done()


session_dict = defaultdict(list)
work_queue = janus.Queue()
workers = []
for worker_id in worker_ids:
    t = Thread(target=worker_loop, daemon=True, args=(worker_id, work_queue.sync_q,))
    t.start()
    workers.append(t)

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse("/docs")


# @app.post("/from_model")
# async def from_embedded_model(
#     model: Annotated[UploadFile, File()],
#     x: Annotated[int, Form()],
#     y: Annotated[int, Form()],
#     img_x: Annotated[int, Form()],
#     img_y: Annotated[int, Form()]
# ):
#     """accepts an embedded image model, coordinates and returns a mask"""
#     logger.info('Started from model route ...')
#     file_data = await model.read()
#     arr_bytes = base64.b64decode(file_data)
#     embedding = sam.buffer_to_embedding(arr_bytes)
#     logger.info('embedded image loaded from POST input ...')

#     # pass everything to the prediction method
#     mask_image = sam.predict_from_embedded(
#             embedding,
#             [x, y],
#             [img_y,img_x]
#     )
#     logger.info('mask returned from predictor ...')

#     # return the mask.
#     file_stream = BytesIO(mask_image)
#     logger.info('file_stream ready to send ...')
#     return StreamingResponse(iter(lambda: file_stream.read(4096), b""), media_type="image/png")


@app.post("/embedded_model", response_class=PlainTextResponse)
async def embedded_model(
    image: Annotated[UploadFile, File()],
    encoding: str = Query("none", description="compress: Response compressed with gzip"),
    session_id: str = Query("none", description="UUID identifying a client"),
    purge_pending: bool = Query(False, description="Purge pending requests for this client")
):
    """Accepts an input image and returns a segment_anything box model
    """
    logger.debug('Started route for embedded_model')
    file_data = await image.read()
    img = utils.buffer_to_image(file_data)

    def do_work(sam):
        return sam.get_box_model(img)

    result_queue = janus.Queue()
    work_item = WorkItem(work_function=do_work, 
                         result_queue=result_queue.sync_q,
                         session_id=session_id)

    if session_id and purge_pending:
        for work_item in session_dict[session_id]:
            logger.warning(f"Marking work item as stale for session {session_id}")
            work_item.stale = True

    session_dict[session_id].append(work_item)

    logger.debug("Putting work function on the work queue")
    await work_queue.async_q.put(work_item)

    logger.debug("Waiting for embedding to be completed by worker...")
    box_model = await result_queue.async_q.get()
    logger.debug('Embedding received')

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logger.debug("Embedding encoded to base64 string")

    if encoding != 'compress':
        logger.debug('Returning uncompressed embedding')
        return b64_string

    logger.debug('Compressing embedding...')
    compressed_data = gzip.compress(b64_bytes)
    headers = {
      "Content-Type": "application/gzip",
      "Content-Encoding": "gzip",
    }
    
    logger.debug('Returning compressed embedding')
    return Response(content=compressed_data, headers=headers)


# @app.post("/prediction")
# async def predict_form(
#     image: Annotated[UploadFile, File()],
#     x: Annotated[int, Form()],
#     y: Annotated[int, Form()],
# ):
#     """accepts an input image and coordinates and returns a prediction mask"""
#     logger.info('Started prediction route ...')
#     file_data = await image.read()
#     logger.info('image loaded from POST input ...')
#     img = sam.buffer_to_image(file_data)
#     mask_image = sam.predict(img, [x, y])
#     logger.info('mask returned from predictor ...')
#     file_stream = BytesIO(mask_image)
#     logger.info('file_stream ready to send ...')
#     return StreamingResponse(iter(lambda: file_stream.read(4096), b""), media_type="image/png")
