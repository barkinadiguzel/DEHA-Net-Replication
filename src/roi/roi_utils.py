import numpy as np

def crop_roi(img, bbox, margin=10):
    x1, y1, x2, y2 = bbox
    h, w = img.shape

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return img[y1:y2, x1:x2]
