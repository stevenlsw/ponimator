import numpy as np
import os
import pickle
from torch.utils import data
from tqdm import tqdm

from .buddi_utils import buddi_style_motion_process


class Buddi_Dataset(data.Dataset):
    def __init__(self, buddi_dir, smplx_layer, device, names=None):
        self.buddi_dir = buddi_dir
        self.result_path = os.path.join(buddi_dir, "buddi_result.pkl")
        self.smplx_layer = smplx_layer
        self.device = device
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        with open(self.result_path, 'rb') as f:
            meta_data = pickle.load(f)
        motion_data = buddi_style_motion_process(meta_data, self.smplx_layer, self.device)
        name = self.buddi_dir.split("/")[-1]
        motion_data["name"] = name
        return motion_data

