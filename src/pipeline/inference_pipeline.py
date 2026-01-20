import torch
import numpy as np
from model.deha_net import DEHANet
from consensus.consensus_module import ConsensusModule
from roi.adaptive_roi import AdaptiveROI
from roi.roi_utils import crop_roi


class InferencePipeline:
    def __init__(self, model_path, device="cuda"):
        self.model = DEHANet()
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.consensus = ConsensusModule(self.model, device=device)
        self.roi = AdaptiveROI()

    def run(self, volume)
        pred_volume = self.consensus.run(volume)

        bbox = self.roi.generate_roi(pred_volume)

        roi_volume = crop_roi(volume, bbox) if bbox else None

        return pred_volume, roi_volume
