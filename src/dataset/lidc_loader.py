import os
import numpy as np
import pydicom

class LIDCLoader:
    def __init__(self, root):
        self.root = root
        self.files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        return img
