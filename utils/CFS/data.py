from torch.utils.data import Dataset
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]