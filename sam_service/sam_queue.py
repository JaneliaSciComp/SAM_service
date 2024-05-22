import os
import sys
import base64
import json
import gzip
import uuid
from asyncio import Lock
from collections import defaultdict
from threading import Thread
from dataclasses import dataclass
from typing import Annotated, Optional, Callable

from fastapi import FastAPI, File, Form, UploadFile, Response, Query
from fastapi.responses import RedirectResponse, PlainTextResponse

import torch
import janus
from loguru import logger

from sam import SAM
import utils

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Configure logging
log_level = config.get('LOG_LEVEL', "DEBUG")
logger.remove()
logger.add(sys.stderr, enqueue=True, level=log_level)

# Get configuration parameters
model_type = config.get('MODEL_TYPE', "vit_h")
module_dir = os.path.dirname(__file__)
checkpoint_file = config.get('CHECKPOINT_FILE', "sam_vit_h_4b8939.pth")
checkpoint = os.path.join(module_dir, checkpoint_file)
gpus = config.get('GPUS', [])
gpu_count = len(gpus)

# Check if a CUDA is available
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
    cancelled: bool = False
    def __str__(self):
        return f"WorkItem(request_id={self.request_id}, session_id={self.session_id}, cancelled={self.cancelled})"


def worker_loop(worker_id:int, work_queue:janus.SyncQueue[WorkItem]):
    """ Loops forever and pulls work from the given work queue.
    """
    worker_device = device
    if worker_device=='cuda' and gpu_count > 1:
        worker_device = f'cuda:{str(gpus[worker_id % gpu_count])}'

    logger.info(f"Creating worker thread {worker_id} running on {worker_device} ...")
    sam = SAM(worker_device, model_type, checkpoint)
    logger.info(f"Worker thread {worker_id} ready")

    while True:
        item = work_queue.get()
        if item.cancelled:
            logger.info(f"Worker {worker_id} ignoring cancelled {item}")
            item.result_queue.sync_q.put(None)
        else:
            result = item.work_function(sam)
            logger.info(f"Worker {worker_id} processed {item}")
            item.result_queue.sync_q.put(result)
        work_queue.task_done()


# Global state that keeps track of work
session_dict = defaultdict(list)
work_queue = janus.Queue()
counter_lock = Lock()
counter = 0

async def get_request_id():
    global counter
    request_id = None
    async with counter_lock:
        request_id = counter
        counter += 1
    return request_id

# Start a worker thread for each GPU
worker_ids = gpus or [0]
logger.info(f"Creating {len(worker_ids)} workers backed by {gpu_count} GPUs")
for worker_id in worker_ids:
    t = Thread(target=worker_loop, daemon=True, args=(worker_id, work_queue.sync_q,))
    t.start()

# Create the API
app = FastAPI(
    title="SAM Service",
    license_info={
        "name": "Janelia Open-Source Software License",
        "url": "https://www.janelia.org/open-science/software-licensing",
    },
)

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse("/docs")


@app.get("/new_session_id", response_class=PlainTextResponse)
async def new_session_id():
    """ Create a new session identifier that can be used to identify your client 
        in other endpoints. 
    """
    # uuid1 creates time-based identifiers which guarantee uniqueness
    return uuid.uuid4()


async def cancel_pending_work_items(request_id, session_id):
    """ Cancel all pending work items for the given session.
    """
    for item in session_dict[session_id]:
        logger.debug(f"R{request_id} - Marking {item} as cancelled")
        item.cancelled = True
        # Send notification to waiting request
        await item.result_queue.async_q.put(None)


async def get_embedding(img, request_id, session_id):
    """ Asynchronously returns the embedding for the given image. 
        This function creates a work item that is executed on a GPU by a worker thread.
        It also handles registering the work item in a global dictionary so that a 
        session's queued items can be invalidated at any time.
    """
    # This result queue is used to communicate the result from the worker thread 
    # back to this function. We only expect one item to be put on this queue.
    result_queue = janus.Queue()

    work_item = WorkItem(request_id=request_id,
                         session_id=session_id,
                         work_function=lambda sam: sam.get_box_model(img), 
                         result_queue=result_queue)
    
    if session_id:
        logger.trace(f"R{request_id} - Adding {work_item} to session dict")
        session_dict[session_id].append(work_item)

    logger.trace(f"R{request_id} - Putting work function on the work queue")
    await work_queue.async_q.put(work_item)

    try:
        logger.debug(f"R{request_id} - Waiting for embedding to be completed by worker ...")
        return await result_queue.async_q.get()
    finally:
        if session_id:
            logger.trace(f"R{request_id} - Removing {work_item} from session dict")
            session_dict[session_id].remove(work_item)
            # Remove the session if there are no items left, to avoid memory leaks
            if len(session_dict[session_id])==0:
                del session_dict[session_id]


@app.post("/embedded_model", response_class=PlainTextResponse)
async def embedded_model(
    image: Annotated[UploadFile, File()],    
    session_id: Optional[str] = Form(None, description="UUID identifying the session"),
    cancel_pending: Optional[bool] = Form(False, description="Cancel any pending requests for this session before processing this one"),
    encoding: str = Query("none", description="compress: Response compressed with gzip"),
):
    """ Accepts an input image and returns a segment_anything box model.
        Optionally also cancel any pending requests from the same session. 
    """
    request_id = await get_request_id()
    logger.debug(f"R{request_id} - Started embedded_model for {session_id}")

    if session_id and cancel_pending:
        await cancel_pending_work_items(request_id, session_id)

    logger.trace(f"R{request_id} - Reading image")
    file_data = await image.read()
    img = utils.buffer_to_image(file_data)

    embedding = await get_embedding(img, request_id, session_id)

    if embedding is None:
        logger.debug(f"R{request_id} - Returning code 499 Client Closed Request")
        return Response(status_code=499, headers={
            "Cancelled-By-Client": "1"
        })

    logger.trace(f"R{request_id} - Computed embedding")

    # Serialize the embedding as base64 string
    arr_bytes = embedding.tobytes()
    b64_bytes = base64.b64encode(arr_bytes)
    b64_string = b64_bytes.decode('utf-8')

    if encoding != 'compress':
        logger.debug(f"R{request_id} - Returning uncompressed embedding")
        return b64_string

    # Compress embedding with GZIP
    logger.trace('Compressing embedding ...')
    compressed_data = gzip.compress(b64_bytes)
    logger.debug(f"R{request_id} - Returning compressed embedding")
    return Response(content=compressed_data, headers={
        "Content-Type": "application/gzip",
        "Content-Encoding": "gzip"
    })


@app.post("/cancel_pending", response_class=PlainTextResponse)
async def cancel_pending(
    session_id: str = Form(..., description="UUID identifying a session")
):
    """Cancel any pending requests for the given session. 
    """
    request_id = await get_request_id()
    await cancel_pending_work_items(request_id, session_id)

