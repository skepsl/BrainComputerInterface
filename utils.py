import torch
from torch.utils.data import Dataset
from einops.layers.torch import Rearrange
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx], dtype=torch.float32)

        label = np.argmax(self.y[idx])
        label = torch.tensor(label, dtype=torch.long)
        return data, label


def epoch_time(start_time, end_time):
    """
    Calculate the time spent to train one epoch
    Args:
        start_time: (float) training start time
        end_time: (float) training end time

    Returns:
        (int) elapsed_mins and elapsed_sec spent for one epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    sample = np.linspace(start=1, stop=409600, num=409600)
    sample = sample.reshape((10, 640, 64))
    sample = sample.transpose(0, 2, 1)

    label = torch.zeros((10,1))
    dataset = MyDataset(sample, label)
    trainer = DataLoader(dataset, shuffle=False, batch_size=2)
    for batch in trainer:
        x, y = batch
        print(x.shape)
