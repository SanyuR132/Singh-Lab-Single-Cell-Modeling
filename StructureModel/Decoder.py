'''
Description:
    Decoder of the VAE.

Author:
    Jiaqi Zhang
'''
import numpy as np
import torch
import torch.nn as nn


class Decoder(nn.Module):
    '''
    Description:
        Decoder module of VAE.
    '''

    def __init__(self, num_layers, layer_size_list):
        '''
        Description:
            Initialization with multiple dense layers.

        Args:
            num_layers (int): The number of dense layers in the decoder.
            layer_size_list (list[int]): Size of each encoder layer.
        '''
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList()

        for i in np.arange(num_layers, 0, -1):
            self.decoders.append(nn.Linear(layer_size_list[i], layer_size_list[i - 1]))


    def forward(self, sample):
        '''
        Description:
            Forward passing for the decoder.

        Args:
            sample (numpy.ndarray): Latent variables.

        Return:
            (numpy.ndarray): Reconstructed data.
        '''
        y = sample
        for i in range(len(self.decoders) - 1):
            y = self.decoders[i](y)
            y = torch.tanh(y)
        # -----
        y = self.decoders[-1](y)
        y = torch.exp(y)
        return y
