import cv2
import numpy as np

def overlay_mask(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = (mask > 0.5).astype(np.uint8) * 255
    img[:, :, 1] = np.maximum(img[:, :, 1], mask)
    return img
