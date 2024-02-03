import os
import base64
import json
import gzip
from asyncio import Lock
from collections import defaultdict
from threading import Thread
from dataclasses import dataclass
from typing import Annotated, Optional, Callable

from fastapi import FastAPI, File, Form, UploadFile, Response, Query
from fastapi.responses import StreamingResponse, RedirectResponse, PlainTextResponse
from loguru import logger
import torch
import janus

from sam import SAM
import utils

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
    request_id: int
    session_id: str
    work_function: Callable
    result_queue: janus.SyncQueue
    stale: bool = False
    def __str__(self):
        return f"WorkItem(request_id={self.request_id}, session_id={self.session_id}, stale={self.stale})"

logger.info(f"Creating {len(worker_ids)} workers backed by {gpu_count} GPUs")

def worker_loop(worker_id:int, work_queue:janus.SyncQueue[WorkItem]):
    """ Loops forever and pulls work from the given work queue.
    """
    worker_device = device
    if worker_device=='cuda' and gpu_count > 1:
        worker_device = f'cuda:{str(gpus[worker_id % gpu_count])}'
    logger.info(f"Creating worker thread {worker_id} running on {worker_device}...")
    sam = SAM(worker_device, model_type, checkpoint)
    logger.info(f"Worker thread {worker_id} ready")
    while True:
        item = work_queue.get()
        if item.stale:
            logger.warning(f"Ignoring {item}")
            item.result_queue.put(None)
        else:
            logger.info(f"Worker {worker_id} processing {item}")
            result = item.work_function(sam)
            logger.info(f"Worker {worker_id} processed {item}")
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


app = FastAPI()
counter_lock = Lock()
counter = 0

@app.post("/embedded_model", response_class=PlainTextResponse)
async def embedded_model(
    image: Annotated[UploadFile, File()],    
    session_id: str = Form(description="UUID identifying a client"),
    purge_pending: Optional[bool] = Form(False, description="Purge pending requests for this client"),
    encoding: str = Query("none", description="compress: Response compressed with gzip"),
):
    """Accepts an input image and returns a segment_anything box model
    """
    global counter
    request_id = None
    async with counter_lock:
        request_id = counter
        counter += 1

    logger.debug(f"R{request_id} - Started route for embedded_model")

    if session_id and purge_pending:
        for item in session_dict[session_id]:
            logger.warning(f"R{request_id} - Marking {item} as stale")
            item.stale = True

    logger.trace(f"R{request_id} - Reading image")
    file_data = await image.read()
    img = utils.buffer_to_image(file_data)

    def do_work(sam):
        return sam.get_box_model(img)

    result_queue = janus.Queue()
    work_item = WorkItem(request_id=request_id,
                         session_id=session_id,
                         work_function=do_work, 
                         result_queue=result_queue.sync_q)
    
    logger.trace(f"R{request_id} - Adding {work_item} to session dict")
    session_dict[session_id].append(work_item)

    logger.trace(f"R{request_id} - Putting work function on the work queue")
    await work_queue.async_q.put(work_item)

    logger.debug(f"R{request_id} - Waiting for embedding to be completed by worker...")
    box_model = await result_queue.async_q.get()

    logger.trace(f"R{request_id} - Removing {work_item} from session dict")
    session_dict[session_id].remove(work_item)

    if box_model is None:
        logger.debug(f"R{request_id} - Returning code 499 Client Closed Request")
        return Response(status_code=499)

    logger.trace(f"R{request_id} - Got embedding")

    # return the model as base64 string.
    arr_bytes = box_model.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')
    logger.trace(f"Embedding encoded to base64 string")

    if encoding != 'compress':
        logger.debug(f"R{request_id} - Returning uncompressed embedding")
        return b64_string

    logger.trace('Compressing embedding...')
    compressed_data = gzip.compress(b64_bytes)
    headers = {
      "Content-Type": "application/gzip",
      "Content-Encoding": "gzip",
    }
    
    logger.debug(f"R{request_id} - Returning compressed embedding")
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
