import torch
import numpy as np
from model.deha_net import DEHANet
from roi.adaptive_roi import AdaptiveROI
from roi.roi_utils import crop_roi

class InferencePipeline:
    def __init__(self, model_path):
        self.model = DEHANet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.roi = AdaptiveROI()

    def run(self, img):
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(x, x).squeeze().numpy()

        bbox = self.roi.generate_roi(pred)
        roi_img = crop_roi(img, bbox) if bbox else None

        return pred, roi_img
