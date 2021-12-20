import mne
import os
import numpy as np
from einops.layers.torch import Rearrange
import torch
from sklearn.preprocessing import StandardScaler


# parent_dir = 'C:/Users/user/PycharmProjects/1_DATASET/physionet_complete'
# fn0 = 'imag_both_feet.set'
# fn1 = 'imag_both_fist.set'
# fn2 = 'imag_left_fist.set'
# fn3 = 'imag_right_fist.set'
# fn4 = 'real_both_feet.set'
# fn5 = 'real_both_fist.set'
# fn6 = 'real_left_fist.set'
# fn7 = 'real_right_fist.set'
#
# EEG0 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn0)))
# EEG1 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn1)))
# EEG2 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn2)))
# EEG3 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn3)))
# EEG4 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn4)))
# EEG5 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn5)))
# EEG6 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn6)))
# EEG7 = np.array(mne.io.read_epochs_eeglab(os.path.join(parent_dir, fn7)))
#
# l0 = np.full(EEG0.shape[0], 0, dtype=np.compat.long)
# l1 = np.full(EEG1.shape[0], 1, dtype=np.compat.long)
# l2 = np.full(EEG2.shape[0], 2, dtype=np.compat.long)
# l3 = np.full(EEG3.shape[0], 3, dtype=np.compat.long)
# l4 = np.full(EEG4.shape[0], 4, dtype=np.compat.long)
# l5 = np.full(EEG5.shape[0], 5, dtype=np.compat.long)
# l6 = np.full(EEG6.shape[0], 6, dtype=np.compat.long)
# l7 = np.full(EEG7.shape[0], 7, dtype=np.compat.long)
#
# EEG = np.concatenate((EEG0, EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7), axis=0)
# label = np.concatenate((l0, l1, l2, l3, l4, l5, l6, l7))
#
# EEG = torch.tensor(EEG, dtype=torch.float32)
# label = torch.tensor(label, dtype=torch.long)
#
# torch.save(EEG, 'C:/Users/user/PycharmProjects/1_DATASET/physionet_complete/physionet_dataset.pt')
# torch.save(label, 'C:/Users/user/PycharmProjects/1_DATASET/physionet_complete/physionet_label.pt')

#
# def window_rearrange(d, window):  # Input shall be [b, c, s]
#     b = np.empty((10, 11, window))
#
#     b[0, 4:7] = d[21:24]
#     b[0, 0:4], b[0, 4:7], b[0, 7:11] = [[0], [0], [0], [0]], d[21:24], [[0], [0], [0], [0]]  # 0
#     b[1, 0:3], b[1, 3:8], b[1, 8:11] = [[0], [0], [0]], d[24:29], [[0], [0], [0]]
#     b[2, 0], b[2, 1:10], b[2, 10] = 0, d[29:38], 0
#     b[3, 0], b[3, 1], b[3, 2:9], b[3, 9], b[3, 10] = 0, d[38], d[0:7], d[39], 0  # 3
#     b[4, 0], b[4, 1], b[4, 2:9], b[4, 9], b[4, 10] = d[42], d[40], d[7:14], d[41], d[43]
#     b[5, 0], b[5, 1], b[5, 2:9], b[5, 9], b[5, 10] = 0, d[44], d[14:21], d[45], 0
#     b[6, 0], b[6, 1:10], b[6, 10] = 0, d[46:55], 0
#     b[7, 0:3], b[7, 3:8], b[7, 8:11] = [[0], [0], [0]], d[55:60], [[0], [0], [0]]
#     b[8, 0:4], b[8, 4:7], b[8, 7:11] = [[0], [0], [0], [0]], d[60:63], [[0], [0], [0], [0]]
#     b[9, 0:5], b[9, 5], b[9, 6:11] = [[0], [0], [0], [0], [0]], d[63], [[0], [0], [0], [0], [0]]
#
#     return b
#
#
# def rearrange(data):
#     b, c, s = data.shape
#     collection = np.empty((b, int(s / 10), 10, 11, 10))
#     for batch in range(b):
#         for seq in range(int(s / 10)):
#             windowed = data[batch, :, seq * 10:(seq + 1) * 10]
#             buffer = window_rearrange(windowed, 10)
#             collection[batch, seq] = buffer
#     return collection


# sample = np.linspace(start=1, stop=40960, num=40960)
# sample = sample.reshape((1, 640, 64))
# sample = sample.transpose(0, 2, 1)
# print(sample[0, :, 0])
# x = rearrange(sample)
#
# print(x.shape)
# snippet = x[0, 0, :, :, 1]
# print(snippet)

# a=[[2,2,2,2,2,2,2,2,2,2],[1,1,1,1,1,1,1,1,1,1]]
# b=[[3,3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4,4]]
# c = np.concatenate((a,b), axis=0)
# scaler = StandardScaler()
#
# scaler.fit(c)
# # scaler.fit(b)
#
# print(scaler.mean_)
#
# def window_rearrange(d, window):  # Input shall be [b, c, s]
#     b = np.empty((10, 11, window))
#
#     b[0, 4:7] = d[21:24]
#     b[0, 0:4], b[0, 4:7], b[0, 7:11] = [[0], [0], [0], [0]], d[21:24], [[0], [0], [0], [0]]  # 0
#     b[1, 0:3], b[1, 3:8], b[1, 8:11] = [[0], [0], [0]], d[24:29], [[0], [0], [0]]
#     b[2, 0], b[2, 1:10], b[2, 10] = 0, d[29:38], 0
#     b[3, 0], b[3, 1], b[3, 2:9], b[3, 9], b[3, 10] = 0, d[38], d[0:7], d[39], 0  # 3
#     b[4, 0], b[4, 1], b[4, 2:9], b[4, 9], b[4, 10] = d[42], d[40], d[7:14], d[41], d[43]
#     b[5, 0], b[5, 1], b[5, 2:9], b[5, 9], b[5, 10] = 0, d[44], d[14:21], d[45], 0
#     b[6, 0], b[6, 1:10], b[6, 10] = 0, d[46:55], 0
#     b[7, 0:3], b[7, 3:8], b[7, 8:11] = [[0], [0], [0]], d[55:60], [[0], [0], [0]]
#     b[8, 0:4], b[8, 4:7], b[8, 7:11] = [[0], [0], [0], [0]], d[60:63], [[0], [0], [0], [0]]
#     b[9, 0:5], b[9, 5], b[9, 6:11] = [[0], [0], [0], [0], [0]], d[63], [[0], [0], [0], [0], [0]]
#     return b
#
# def rearrange(snippet):
#     b, c, s = snippet.shape
#     target = np.empty((b,int(s / 10), 10, 10, 11))
#     for batch in range(b):
#         collection = np.empty((int(s / 10), 10, 11, 10), dtype='float32')
#         for seq in range(int(s / 10)):
#             windowed = snippet[batch, :, seq * 10:(seq + 1) * 10]
#             buffer = window_rearrange(windowed, 10)
#             collection[seq] = buffer
#         collection = collection.transpose(0, 3, 1, 2)
#         target[batch]=collection
#
#     return target
#
# parent_dir = 'C:/Users/user/PycharmProjects/1_DATASET/physionet_complete/EEG'
# file_name = 'physionet_dataset.pt'
#
# x_train, y_train, x_val, y_val = torch.load(os.path.join(parent_dir, file_name))
#
# dataset = torch.cat((x_train, x_val), dim=0)
#
# mean = dataset.mean()
# std = dataset.std()
#
# data = [mean, std]
# dataset = (dataset-mean)/std
#
# print(dataset.max())
# print(dataset.min())
# print(dataset.mean())
# print(dataset.std())
#
# torch.save(data, 'norm_param.pt')
# # print(mean)
# # print(std)
#
#

label =[]
label.append([0,1,0])
label.append([0,1,0])
label.append([0,1,0])
label.append([0,1,0])


print(set(label))















