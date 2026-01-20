import numpy as np

class AdaptiveROI:
    def __init__(self, R_T=0.6):
        self.R_T = R_T

    def generate_roi(self, prob_map):
        mask = prob_map > self.R_T
        coords = np.argwhere(mask)

        if len(coords) == 0:
            return None

        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)

        return (x1, y1, x2, y2)
