import os
import sys
import tempfile
import atexit
import shutil
from loguru import logger
import warnings

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import paintera_sam.sam_model as model
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

import cv2
import numpy as np
import torch

@atexit.register
def cleanup_temp_dir():
    try:
        shutil.rmtree(temp_dir)
    except IOError:
        sys.stderr.write('Failed to clean up temp dir {}'.format(temp_dir))

temp_dir = tempfile.mkdtemp()

class SAM:

    def __init__(self, device, model_type, checkpoint):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=device)

        logger.debug(f"Creating new predictor for {model_type} on {device}...")
        self.sam_predictor = SamPredictor(self.sam)
        logger.debug("Created new predictor")

        #self.ort_session = self.create_onnx_runtime(model_type, checkpoint)

            
    def create_onnx_runtime(self, model_type, checkpoint):
        logger.debug(f"Creating new ONNX runtime for {model_type}...")

        # there needs to be a seperate onnx_sam model that is sent to the cpu,
        # because the onnx export occurs on the cpu and not the gpu.
        onnx_sam = sam_model_registry[model_type](checkpoint=checkpoint)
        onnx_sam.to(device='cpu')

        onnx_model_path = os.path.join(temp_dir, f"{str(os.getpid())}.sam_onnx.onnx")

        onnx_model = SamOnnxModel(onnx_sam, return_single_mask=True)

        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

        embed_dim = self.sam.prompt_encoder.embed_dim
        embed_size = self.sam.prompt_encoder.image_embedding_size
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

        onnx_model_quantized_path = os.path.join(temp_dir, f"{str(os.getpid())}.sam_onnx_quantized_example.onnx")

        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=onnx_model_quantized_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        onnx_model_path = onnx_model_quantized_path
        logger.debug('Created new ONNX runtime')

        return onnxruntime.InferenceSession(onnx_model_path)


    def predict(self, cv_image, coords):
        logger.trace(f"Running prediction on {self.device}")
        self.sam_predictor.set_image(cv_image, image_format='BGR')
        masks = model.predict_current_image(self.sam_predictor, *coords, cv_image, show=False)
        _, buffer = cv2.imencode('.png', masks[0] * 255)
        return buffer


    def predict_from_embedded(self, image_embedding, coords, img_dimensions):
        logger.trace(f"Running embedded prediction on {self.device}")
        input_point = np.array([coords])
        input_label = np.array([1])
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = self.sam_predictor.transform.apply_coords(onnx_coord, img_dimensions).astype(np.float32)

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

        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = masks > self.sam_predictor.model.mask_threshold

        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * 255
        _, buffer = cv2.imencode('.png', mask_image)
        return buffer


    def get_box_model(self, cv_image):
        logger.trace(f"Embedding image on {self.device}")
        self.sam_predictor.set_image(cv_image, image_format='BGR')
        return self.sam_predictor.get_image_embedding().cpu().numpy()
