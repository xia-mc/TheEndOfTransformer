import torch
from torch.utils.data import Dataset

from Const import DTYPE


class ModelDataSet(Dataset):
    def __init__(self, inData, outData):
        self.inData = torch.tensor(inData, dtype=DTYPE)
        self.outData = torch.tensor(outData, dtype=DTYPE)

    def __len__(self):
        return len(self.inData)

    def __getitem__(self, idx):
        return self.inData[idx], self.outData[idx]
