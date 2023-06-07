"""
Description:
    Utility functions for Python scrips.

Author:
    Jiaqi Zhang
"""

from datetime import datetime

import scanpy
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

# -------------------------------------------------


class CustomDataset(Dataset):
    """
    Description:
        Class for loading dataset.
    """

    def __init__(self, x, y):
        """
        Initialization. Since it is used for VAE reconstruction, x and y are the same.
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return (self.x[index, :], self.y[index, :])

    def __len__(self):
        return len(self.x)


# -------------------------------------------------


def to_cpu_npy(x):
    """
    Description:
        Extract data from GPU to CPU.
    """
    if type(x) == list:
        new_x = []
        for element in x:
            new_x.append(element.detach().cpu().numpy())
    else:
        new_x = x.detach().cpu().numpy()
    return new_x


def readH5ad(filename):
    """
    Description:
        Read h5ad file.
    """
    data = scanpy.read_h5ad(filename)
    return data.to_df().values


def readMtx(filename, transpose=True):
    sparse_mat = mmread(filename)  # coo format
    if transpose:
        # rows now correspond to samples
        sparse_mat = coo_matrix.transpose(sparse_mat)
    sparse_mat = coo_matrix.tocsr(sparse_mat)  # easier to index
    dense_mat = np.array(csr_matrix.todense(sparse_mat))
    return dense_mat


# -------------------------------------------------


def prepareDataset(train_set, valid_set, test_set, batch_size):
    # read expression matrix
    # train_set = _readH5ad(train_filename)
    # valid_set = _readH5ad(validate_filename)
    # test_set = _readH5ad(test_filename)
    print("Training data shape : ", train_set.shape)
    print("Validate data shape : ", valid_set.shape)
    print("Testing data shape : ", test_set.shape)
    # construct the DataLoader
    train_set = DataLoader(
        dataset=CustomDataset(train_set, train_set), batch_size=batch_size, shuffle=True
    )
    valid_set = DataLoader(
        dataset=CustomDataset(valid_set, valid_set), batch_size=batch_size, shuffle=True
    )
    test_set = DataLoader(
        dataset=CustomDataset(test_set, test_set), batch_size=batch_size, shuffle=True
    )
    return train_set, valid_set, test_set


def prepareAugmentDataset(data_set, batch_size, train_size=0.7):
    # read expression matrix
    # data_set = _readH5ad(filename)
    sampled_data, removed_data = train_test_split(
        data_set, train_size=train_size, shuffle=True
    )
    print("Original data shape : ", data_set.shape)
    print("Down-sampled data shape : ", sampled_data.shape)
    print("Removed data shape : ", removed_data.shape)
    # construct the DataLoader
    sampled_set = DataLoader(
        dataset=CustomDataset(sampled_data, sampled_data),
        batch_size=batch_size,
        shuffle=True,
    )
    return sampled_set, sampled_data, removed_data


# -------------------------------------------------


def getTimeStr():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------------------------------


def addDefaultArg(config_dict):
    """
    Description:
        Add default arguments for model running.
    """
    if "layer_size_list" not in config_dict or "num_layers" not in config_dict:
        raise ValueError("Must define layer size before run the model!")
    if "max_epoch" not in config_dict:
        config_dict["max_epoch"] = 50
    if "batch_size" not in config_dict:
        config_dict["batch_size"] = 128
    if "beta" not in config_dict:
        config_dict["beta"] = 0.5
    if "cluster_weight_type" not in config_dict:
        config_dict["cluster_weight_type"] = "sigmoid"
    if "learning_rate" not in config_dict:
        config_dict["learning_rate"] = 1e-4
    if "need_save" not in config_dict:
        config_dict["need_save"] = True
    return config_dict


def getDefualtLayerSize(num_genes, num_layers, latent_size):
    a = num_genes
    b = -np.log2(num_genes / latent_size) / num_layers
    layer_size_list = [num_genes]
    for i in np.arange(1, num_layers, 1):
        new_layer_size = int(a * 2 ** (b * i))
        layer_size_list.append(new_layer_size)
    layer_size_list.append(latent_size)
    return layer_size_list


if __name__ == "__main__":
    default_args = addDefaultArg({})
    getTimeStr()
