'''
Description:
    Encoder of the VAE.

Author:
    Jiaqi Zhang
'''
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    '''
    Description:
        Encoder module of VAE.
    '''

    def __init__(self, num_layers, layer_size_list, cluster_weight_type):
        '''
        Description:
            Initialization of encoder.

        Args:
            num_layers (int): The number of dense layers in the decoder.
            layer_size_list (list[int]): Size of each encoder layer.
            cluster_weight_type (str): Specify the type of weights for clusters.
                                       Should be one of "vanilla", "softmax", or "sigmoid".
        '''
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList()
        self.num_keypoints = layer_size_list[-1]
        # -----
        for i in np.arange(0, num_layers - 1, 1):
            self.encoders.append(nn.Linear(layer_size_list[i], layer_size_list[i + 1]))
        self.encoders.append(nn.Linear(layer_size_list[-2], 3 * layer_size_list[-1]))
        self.softmax_func = nn.Softmax(dim=-1)
        self.cluster_weight_type = cluster_weight_type


    def forward(self, x):
        '''
        Description:
            Forward passing for the decoder.

        Args:
            x (numpy.ndarray): A batch of training data.

        Return:
            (numpy.ndarray): k_mu
            (numpy.ndarray): k_logvar
            (numpy.ndarray): k_weights
        '''
        k = x
        ### there are num_layers + 1 encoder layers b/c Gaussian has two parameters
        for i in range(len(self.encoders) - 1):
            k = self.encoders[i](k)
            k = torch.tanh(k)
        k = self.encoders[-1](k)
        k_mu = k[:, :self.num_keypoints]
        k_logvar = k[:, self.num_keypoints:self.num_keypoints * 2]
        k_weights = k[:, self.num_keypoints * 2:self.num_keypoints * 3]
        if self.cluster_weight_type == 'softmax':
            k_weights = self.softmax_func(k_weights)
        elif self.cluster_weight_type == 'sigmoid':
            k_weights = torch.sigmoid(k_weights)
        elif self.cluster_weight_type == 'vanilla':
            k_weights = torch.ones_like(k_weights)
        else:
            k_weights = torch.empty((0))
        return k_mu, k_logvar, k_weights # , k