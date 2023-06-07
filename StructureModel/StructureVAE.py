'''
Description:
    Codes of the VAE model.

Author:
    Jiaqi Zhang
'''
import torch
import torch.nn as nn
from geomloss import SamplesLoss


import sys
sys.path.append("./")
from StructureModel.Encoder import Encoder
from StructureModel.Decoder import Decoder

# from Models.StructureModel.Encoder import Encoder
# from Models.StructureModel.Decoder import Decoder


class VAE(nn.Module):
    '''
    Description:
        VAE model for scRNA-seq data simulation.
    '''

    def __init__(self, num_layers, layer_size_list, cluster_weight_type='vanilla', beta=[0.5, 0.5]):
        '''
        Description:
            Initialize the VAE model.

        Args:
            num_layers (int): The number of layers
            layer_size_list (list): The size of each layer
            cluster_weight_type (str): The type of cell cluster weight (vanilla, sigmoid, or softmax).
            beta (list): The parameter weighting between the reconstruction and KL loss.
        '''
        super(VAE, self).__init__()
        self.encoder_module = Encoder(num_layers, layer_size_list, cluster_weight_type)
        self.decoder_module = Decoder(num_layers, layer_size_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cluster_weight_type = cluster_weight_type
        self.beta = beta # trade-off between reconstruction and KL loss
        # ---
        self.mu = None
        self.logvar = None
        self.z_weights = None


    def forward(self, x):
        '''
        Description:
            Model forward passing.

        Args:
            x (numpy.ndarray)： Input data with the shape of (batch size, num of cells, num of genes)

        Return:
            y (numpy.ndarray): Reonstructed data.
            z_mu (numpy.ndarray): Latent variable for the distribution mean.
            z_logvar (numpy.ndarray): Latent variable for the distribution log variance.
        '''
        # compute latent variables
        k_mu, k_logvar, k_weights = self.encoder_module(x)
        # sampling based on latent variables
        sample = self.latent_sample(k_mu, k_logvar, k_weights)
        # reconstruction from decoder
        y = self.decoder_module(sample)
        # -----
        self.mu = k_mu
        self.logvar = k_logvar
        self.z_weights = k_weights
        return y, k_mu, k_logvar


    def loss(self, x, y, z_mu, z_logvar):
        '''
        Descripiton:
            VAE loss function.

        Args:
            x: Data.
            y: Label.
            z_mu: Latent variable.
            z_logvar: Latent variable.

        Return:
            (float): Loss value.
        '''
        # reconstruction loss
        recon_loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        recon_loss = recon_loss_func(x, y)
        # KL divergence
        KL_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        # weighted summation
        loss_components = [self.beta[0] * recon_loss, self.beta[1] * KL_loss]
        vae_loss = loss_components[0] + loss_components[1]
        return vae_loss



    def latent_sample(self, z_mu, z_logvar, z_weights):
        '''
        Description:
            Sampling latent variables form th Gaussian distribution.
        '''
        std = z_logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        sample = eps.mul(std).add_(z_mu)
        if self.cluster_weight_type != 'vanilla':
            # element-wise multiplication
            sample = torch.mul(sample, z_weights)
        return sample


    def generate(self, x):
        '''
        Description:
            Generate the reconstructed data given a sample.
        '''
        generation = self.forward(x)
        return generation