import numpy as np
import hdf5storage
from torch.utils.data import Dataset
import torch

class Yield(Dataset): 
    def __init__(self, data):
        self.yield_columns = [col for col in data.columns if 'YIELD_' in col]
        self.data = data[self.yield_columns].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the data for the given index
        data = self.data.iloc[idx].values

        # Normalize to [-1, 1] across all features
        # data = 2 * ((data - np.min(data)) / (np.max(data) - np.min(data))) - 1
        data = torch.from_numpy(data.astype('float32').reshape(1, -1))  

        return data

class Weather(Dataset): 
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the data for the given index
        return self.data[idx].reshape(1,-1)
