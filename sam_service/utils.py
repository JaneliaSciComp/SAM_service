import numpy as np
import cv2

def buffer_to_image(buffer):
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def buffer_to_embedding(buffer):
    nparr = np.frombuffer(buffer, dtype=np.float32)
    # restore the orginal shape of the numpy array, because frombuffer will
    # only create a one dimensional array.
    return nparr.reshape((1,256,64,64))
