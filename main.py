
import numpy as np
import torch
import time
import pandas as pd
import os
import warnings
from sklearn.utils import shuffle
import wandb


from utils import MyDataset
from torch.utils.data import DataLoader, TensorDataset
from trainer import Trainer
from utils import epoch_time


def training():
    data_name = '1_108_shuffle_dataset_3D_win_10.pkl'
    label_name = '1_108_shuffle_labels_3D_win_10.pkl'
    file_path = 'D:/Dataset Collection/physioNet/preprocessed\Dalin Zhang'
    window_size = 10

    datasets = np.load(os.path.join(file_path, data_name), allow_pickle=True)
    labels = np.load(os.path.join(file_path, label_name), allow_pickle=True)

    datasets = datasets.reshape((len(datasets), window_size,1, 10, 11))
    labels = np.asarray(pd.get_dummies(labels))

    split = np.random.rand(len(datasets)) < 0.75
    x_train = datasets[split]

    y_train = labels[split]

    x_val = datasets[~split]
    y_val = labels[~split]

    trainer = Trainer()

    train_data = MyDataset(x_train, y_train)
    val_data = MyDataset(x_val, y_val)

    train_iter = DataLoader(train_data, batch_size=300, shuffle=False, num_workers=1)
    val_iter = DataLoader(val_data, batch_size=300, shuffle=False, num_workers=1)

    length = [x_train.shape[0], x_val.shape[0]]

    epochs = 30
    trainer.train(train_iter, val_iter, epochs, length)


if __name__ == '__main__':
    training()





















