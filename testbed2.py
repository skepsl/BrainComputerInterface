
import numpy as np
import os
import pandas as pd


# data_name = '1_108_shuffle_dataset_3D_win_10.pkl'
label_name = '1_108_shuffle_labels_3D_win_10.pkl'
file_path = 'D:/raw_data'

# datasets = np.load(os.path.join(file_path, data_name), allow_pickle=True)
labels = np.load(os.path.join(file_path, label_name), allow_pickle=True)
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print(one_hot_labels)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

print(labels[0])
window_size = 10
# print(datasets.shape)
print(labels.shape)

# datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)

# print(datasets.shape)




