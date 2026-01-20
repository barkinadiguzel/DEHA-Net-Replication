import numpy as np
import torch


class TriPlanarSlicer:
    def __init__(self, volume):
        self.volume = volume

    def axial(self):
        return [self.volume[i, :, :] for i in range(self.volume.shape[0])]

    def coronal(self):
        return [self.volume[:, i, :] for i in range(self.volume.shape[1])]

    def sagittal(self):
        return [self.volume[:, :, i] for i in range(self.volume.shape[2])]


class ConsensusModule:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _predict_slice(self, img):
        x = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x, x)
            pred = pred.squeeze().cpu().numpy()

        return pred

    def _predict_stack(self, slices):
        preds = []
        for s in slices:
            preds.append(self._predict_slice(s))
        return preds

    def run(self, volume):
        slicer = TriPlanarSlicer(volume)

        axial_slices = slicer.axial()
        coronal_slices = slicer.coronal()
        sagittal_slices = slicer.sagittal()

        # Slice inference
        axial_preds = self._predict_stack(axial_slices)
        coronal_preds = self._predict_stack(coronal_slices)
        sagittal_preds = self._predict_stack(sagittal_slices)

        # Reconstruct volumes
        axial_vol = np.stack(axial_preds, axis=0)        
        coronal_vol = np.stack(coronal_preds, axis=1)    
        sagittal_vol = np.stack(sagittal_preds, axis=2)  

        # Consensus fusion
        consensus = (axial_vol + coronal_vol + sagittal_vol) / 3.0

        return consensus
