'''
vae_rnn_detached
v1j: filtered genes so that genes below and above thresholds excluded


v1g: Justin's hyperparameters;
softmax func changed from dim=1 to dim=-1 and with Justin's hyperparameters
v1: vae_rnn_v1 but with dynamics commented out 


vae_rnn
v1: cleaned up version of vrnn_exp_thresh_zprob_v8

(vrnn_exp_thresh_zprob)
v8: use calcB, introduce alpha to weight KL
v7b: (H(X) - H(Y)) - KL(Y|X) - computed in two ways; LSA maximize
v7: revert v5 minus cluster entropy, revise metric to H(X) - KL(Y|X)
v6: use boltzmann dist for sampling with cosine metric - did not go well
v5: use softmax for cluster weights and add in cluster entropy to RNN
v4: VAE: Set thresh=0, added in linear dropout layers 
v3: VRNN loss with KL divergence and MSE/Wasserstein
v2: exp-thresh with dynamics
'''

import os
import argparse
import random
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import cm
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

import scipy as scp
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.special import softmax as scp_softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from statsmodels.distributions.empirical_distribution import ECDF

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import Sampler

from geomloss import SamplesLoss
from six.moves import cPickle as pickle

import umap




def to_cpu_npy(x):
    if type(x) == list:
        new_x = []
        for element in x:
            new_x.append(element.detach().cpu().numpy())
    elif x.is_cuda == True:
        new_x = x.detach().cpu().numpy()
    elif x.is_cuda == False:
        new_x = x.detach().numpy()
    return new_x


def calc_dist(np_k_mat_t, np_k_mat_tp1, latent_size, alpha=0.5):

    dist_list = []
    
    ent_t = np_k_mat_t[:,latent_size:2*latent_size] + np.sqrt(2*np.pi) + 0.5
    ent_tp1 = np_k_mat_tp1[:,latent_size:2*latent_size] + np.sqrt(2*np.pi) + 0.5
    ent_delta = ent_t - ent_tp1
    ent_weighted = np_k_mat_t[:,2*latent_size:]*ent_delta
    ent_weighted_reduce = np.sum(ent_weighted, axis=1)
    
    for i in np.arange(np.size(np_k_mat_t, 0)):

        logvar_t = np_k_mat_t[i,latent_size:2*latent_size]
        logvar_tp1 = np_k_mat_tp1[:,latent_size:2*latent_size] 
        mu_t = np_k_mat_t[i, :latent_size]
        mu_tp1 = np_k_mat_tp1[:, :latent_size]
        KL = logvar_t - logvar_tp1 + (np.exp(2*logvar_tp1) + \
            (mu_t - mu_tp1)**2)/(2*np.exp(2*logvar_tp1)) - 0.5
        KL_weighted = np_k_mat_t[:,2*latent_size:]*KL
        KL_weighted_reduce = np.sum(KL_weighted, axis=1)
        
        metric = alpha*ent_weighted_reduce - (1-alpha)*KL_weighted_reduce
        metric = metric - np.min(metric)
        metric = np.reshape(metric, (1, -1))
        dist_list.append(metric)
        
    dist_mat = np.concatenate(dist_list, axis=0)

    return dist_mat


class StructCustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __getitem__(self, index):
        ### needs to return sample and target label by index
        # return (self.x[index, :], self.x[index, :])
        return (self.x[index, :], self.y[index, :])

    def __len__(self):
        return len(self.x)

    
class Encoder(nn.Module):
    def __init__(self, num_layers, layer_size_list_struct, cluster_weight_type):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList()
        self.num_keypoints = layer_size_list_struct[-1]

        for i in np.arange(0, num_layers - 1, 1):
            self.encoders.append(nn.Linear(layer_size_list_struct[i], layer_size_list_struct[i+1]))
        
        self.encoders.append(nn.Linear(layer_size_list_struct[-2], 3*layer_size_list_struct[-1]))

        # self.z_mu_transform = nn.Linear(layer_size_list_struct[-2], layer_size_list_struct[-1])
        # self.z_logvar_transform = nn.Linear(layer_size_list_struct[-2], layer_size_list_struct[-1])
        # self.z_cluster_weight_transform = nn.Linear(layer_size_list_struct[-2], layer_size_list_struct[-1])

        self.softmax_func = nn.Softmax(dim=-1)
        self.cluster_weight_type = cluster_weight_type

    def forward(self, x):
        k = x

        ### there are num_layers + 1 encoder layers b/c Gaussian has two parameters
        for i in range(len(self.encoders) - 1):

            k = self.encoders[i](k)
            k = torch.tanh(k)

        k = self.encoders[-1](k)
        k_mu = k[:, :self.num_keypoints]
        k_logvar = k[:, self.num_keypoints:self.num_keypoints*2]    
        
        k_weights = k[:, self.num_keypoints*2:self.num_keypoints*3]
        if self.cluster_weight_type == 'softmax':
            k_weights = self.softmax_func(k_weights)
        elif self.cluster_weight_type == 'sigmoid':
            k_weights = torch.sigmoid(k_weights)
        elif self.cluster_weight_type == 'vanilla':
            k_weights = torch.ones_like(k_weights)
        else:
            k_weights = torch.empty((0))
        
        return k_mu, k_logvar, k_weights, k


class Decoder(nn.Module):
    def __init__(self, num_layers, layer_size_list_struct):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList()
            
        for i in np.arange(num_layers, 0, -1):
            self.decoders.append(nn.Linear(layer_size_list_struct[i], layer_size_list_struct[i-1]))
                
    def forward(self, sample):        
        y = sample
        
        for i in range(len(self.decoders) - 1):
            y = self.decoders[i](y)
            y = torch.tanh(y)
            
        y = self.decoders[-1](y)
        y = torch.exp(y)
        
        return y
    

class VRNN(nn.Module):
    def __init__(self, batch_size, hidden_size, keypoint_size, loss_type='wass_weighted'):
        super(VRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keypoint_size = keypoint_size
        self.batch_size = batch_size

        self.hidden_state = torch.empty([1, batch_size, hidden_size*3], device=self.device)
        torch.nn.init.zeros_(self.hidden_state)
        
        self.loss_type = loss_type
        # torch.nn.init.normal_(self.hidden_state)

        self.rnn = nn.GRU(keypoint_size*3, hidden_size*3, batch_first=True)
                             
    def forward(self, x, t, h):
        
        # pi_weights = x[:, :, 2*self.keypoint_size:]
        # entropies = []
        # for i in torch.arange(self.batch_size):
        #     dist = pi_weights[i, :, :].flatten()
        #     entropies.append(torch.distributions.categorical.Categorical(probs=dist).entropy())
        # entropies = torch.tensor(entropies).to(self.device)
        # entropies = entropies.reshape(self.batch_size, 1, 1)
        # x = torch.cat((x, entropies), dim=2)        

        # x = torch.cat((x, t), dim=2)        
        
        if h == None:
            h = self.hidden_state

        _, out = self.rnn(x, h)
                        
        return out
    
    def loss(self, h, y):
        ###https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
                
        h_mu = h[:, :, :self.keypoint_size]
        h_logvar = h[:, :, self.keypoint_size:2*self.keypoint_size]
        h_var = h_logvar.exp()
        h_weights = h[:, :, 2*self.keypoint_size:]

        y_mu = y[:, :, :self.keypoint_size]
        y_logvar = y[:, :, self.keypoint_size:2*self.keypoint_size]
        y_var = y_logvar.exp()
        y_weights = y[:, :, 2*self.keypoint_size:]

        MSE_loss_func = torch.nn.MSELoss(reduction='none')

        if self.loss_type == 'MSE':
            loss = torch.sum(MSE_loss_func(h_mu, y_mu))
        elif self.loss_type == 'MSE_weighted':
            MSE_unweighted = MSE_loss_func(h_mu, y_mu) + \
                MSE_loss_func(h_var, y_var)
            weight_factors = MSE_loss_func(h_weights, y_weights)
            loss = torch.sum(torch.mul(MSE_unweighted, weight_factors))
        elif self.loss_type == 'KL':
            loss = torch.sum(0.5*torch.log(h_var) - 0.5*torch.log(y_var) + \
                (y_var + (y_mu - h_mu)**2)/(2*h_var) - 0.5)
        elif self.loss_type == 'KL_weighted':
            KL_loss = torch.sum(0.5*torch.log(h_var) - 0.5*torch.log(y_var) + \
                (y_var + (y_mu - h_mu)**2)/(2*h_var) - 0.5)
            weight_factors = MSE_loss_func(h_weights, y_weights)
            loss = torch.sum(torch.mul(KL_loss, weight_factors))
        elif self.loss_type == 'wass':
            ### from: https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
            mu_term = torch.sum(MSE_loss_func(h_mu, y_mu))
            Bsq_term = torch.sum(h_var) + torch.sum(y_var) + \
                2*(torch.sum(h_var)*torch.sum(y_var))**(1/2)
            loss = mu_term + Bsq_term
        elif self.loss_type == 'wass_weighted':
            mu_vec = MSE_loss_func(h_mu, y_mu)
            Bsq_vec = h_var + y_var + 2*torch.sqrt((torch.mul(h_var, y_var)))
            weight_factors = MSE_loss_func(h_weights, y_weights)
            loss = torch.sum(torch.mul(weight_factors, mu_vec + Bsq_vec))
        elif self.loss_type == 'MSE_KL_weighted':
            MSE_unweighted = MSE_loss_func(h_mu, y_mu) + \
                MSE_loss_func(h_var, y_var)
            weight_factors = MSE_loss_func(h_weights, y_weights)
            MSE_loss = torch.sum(torch.mul(MSE_unweighted, weight_factors))
            
            KL_loss = torch.sum(0.5*torch.log(h_var) - 0.5*torch.log(y_var) + \
                (y_var + (y_mu - h_mu)**2)/(2*h_var) - 0.5)
            weight_factors = MSE_loss_func(h_weights, y_weights)
            KL_loss = torch.sum(torch.mul(KL_loss, weight_factors))
            
            loss = MSE_loss + KL_loss
        elif self.loss_type == 'Wass_KL_weighted':
            mu_vec = MSE_loss_func(h_mu, y_mu)
            Bsq_vec = h_var + y_var + 2*torch.sqrt((torch.mul(h_var, y_var)))
            weight_factors = MSE_loss_func(h_weights, y_weights)
            wass_loss = torch.sum(torch.mul(weight_factors, mu_vec + Bsq_vec))
            
            KL_loss = torch.sum(0.5*torch.log(h_var) - 0.5*torch.log(y_var) + \
                (y_var + (y_mu - h_mu)**2)/(2*h_var) - 0.5)
            weight_factors = MSE_loss_func(h_weights, y_weights)
            KL_loss = torch.sum(torch.mul(KL_loss, weight_factors))
            
            loss = wass_loss + KL_loss
            
        return loss


class Linear_sparse(nn.Module):
    def __init__(self, num_genes, num_layers=2):
        super(Linear_sparse, self).__init__()
        self.sparse_layers = nn.ModuleList()

        for i in np.arange(0, num_layers, 1):
            self.sparse_layers.append(nn.Linear(num_genes, num_genes))
    
    def forward(self, x):
        mask = x
        
        for i in range(len(self.sparse_layers)): 
            mask = self.sparse_layers[i](mask)
            mask = torch.relu(mask)
        
        mask = torch.sigmoid(mask)
        x = torch.mul(mask >= 0.5, x) + 0
        
        return x
    
    
class VAE_struct(nn.Module):
    def __init__(self, num_layers, layer_size_list_struct, cluster_weight_type='vanilla', beta_struct=[0.5, 0.5]):
        super(VAE_struct, self).__init__()
        self.encoder_module = Encoder(num_layers, layer_size_list_struct, cluster_weight_type)
        self.decoder_module = Decoder(num_layers, layer_size_list_struct)
        # self.dropout_module = Linear_sparse(layer_size_list_struct[0])
        self.cluster_weight_type = cluster_weight_type
        self.beta_struct = beta_struct
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, return_mean=0):
        k_mu, k_logvar, k_weights, k = self.encoder_module(x)
        sample = self.latent_sample(k_mu, k_logvar, k_weights)
        y = self.decoder_module(sample)
        # y = self.dropout_module(y)
        return y, k_mu, k_logvar, k_weights, k
    
    def forward_predict(self, h, latent_size, return_mean=0):
        k_mu = h[:, :latent_size]
        k_logvar = h[:, latent_size:latent_size*2]
        k_weights = h[:, latent_size*2:]
        sample = self.latent_sample(k_mu, k_logvar, k_weights)
        y = self.decoder_module(sample)
        # y = self.dropout_module(y)
        
        return y
    
    def loss(self, x, y, k_mu, k_logvar):
        
        recon_loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        recon_loss = recon_loss_func(x, y)
        
        KL_loss = -0.5*torch.sum(1 + k_logvar - k_mu.pow(2) - k_logvar.exp())
        
        vae_loss = self.beta_struct[0]*recon_loss + self.beta_struct[1]*KL_loss

        vae_components = [self.beta_struct[0]*recon_loss, self.beta_struct[1]*KL_loss]

        return vae_loss, vae_components
    
    def latent_sample(self, k_mu, k_logvar, k_weights, return_mean=0):
            
        if return_mean == 0:
            # the reparameterization trick
            
            std = k_logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            sample = eps.mul(std).add_(k_mu)

        else:
            sample = k_mu
        
        if self.cluster_weight_type != 'vanilla':     
            ###element-wise multiplication
            sample = torch.mul(sample, k_weights)
        
        return sample
    
        
def train_struct_model(model, optimizer, train_set, valid_set, max_epoch, device):

    model = model.to(device)

    train_loss_list = []
    # train_pcc_list = []
    valid_loss_list = []
    # valid_pcc_list = []

    train_recon_loss_list = []
    train_kl_loss_list = []
    valid_recon_loss_list = []
    valid_kl_loss_list = []

    for epoch in np.arange(max_epoch):
        
        print('Epoch ', epoch)

        train_loss = 0
        valid_loss = 0

        model.train()
        train_pred_list = []        
        train_labels_list = []
        
        train_recon_loss = 0
        train_kl_loss = 0
        
        for x_batch, _ in train_set:
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            forward_scores, k_mu, k_logvar, k_weights, k = model(x_batch)
            loss, vae_components_list = model.loss(forward_scores, x_batch, k_mu, k_logvar)
            
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
            
            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            train_pred_list.append(to_cpu_npy(pred))
            train_labels_list.append(to_cpu_npy(x_batch))
        
            [train_recon, train_kl] = [v for v in vae_components_list]            
            train_recon_loss += train_recon.item()
            train_kl_loss += train_kl.item()
        
        train_loss_list.append(train_loss/len(train_labels_list))
        train_recon_loss_list.append(train_recon_loss/len(train_labels_list))
        train_kl_loss_list.append(train_kl_loss/len(train_labels_list))
                
        model.eval()
        
        valid_recon_loss = 0
        valid_kl_loss = 0
        
        with torch.no_grad():

            for x_batch, _ in valid_set:
                x_batch = x_batch.to(device)
                
                forward_scores, k_mu, k_logvar, k_weights, k = model(x_batch)
                                
                loss, vae_components_list = model.loss(forward_scores, x_batch, k_mu, k_logvar)
                valid_loss += loss.item()

                pred = torch.reshape(forward_scores, (len(forward_scores), -1))
                
                [valid_recon, valid_kl] = [v for v in vae_components_list]            
                valid_recon_loss += valid_recon.item()
                valid_kl_loss += valid_kl.item()

        valid_loss_list.append(valid_loss/len(valid_set))
        valid_recon_loss_list.append(valid_recon_loss/len(valid_set))
        valid_kl_loss_list.append(valid_kl_loss/len(valid_set))
        
    return train_loss_list, train_recon_loss_list, train_kl_loss_list, \
        valid_loss_list, valid_recon_loss_list, valid_kl_loss_list


def eval_struct_model(model, test_set, device):
    model = model.to(device)

    model.eval()
    test_loss = 0
    test_pred_list = []
    test_labels_list = []

    with torch.no_grad():
        # for x_batch in test_set:
        for x_batch, _ in test_set:
            x_batch = x_batch.to(device)
                
            forward_scores, k_mu, k_logvar, k_weights, k = model(x_batch)
            
            loss, _ = model.loss(forward_scores, x_batch, k_mu, k_logvar)

            test_loss += loss.item()

            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            test_pred_list.append(to_cpu_npy(pred))
            test_labels_list.append(to_cpu_npy(x_batch))

    test_scores = np.concatenate(test_pred_list)
    test_labels = np.concatenate(test_labels_list)

    return test_loss, test_scores, test_labels, k_mu, k_logvar, \
            k_weights, k


def train_dyn_model(model, optimizer, train_set, valid_set, max_epoch, batch_size, tp_hr_list, device):

    model.to(device)
    
    train_loss_list = []
    valid_loss_list = []
    
    batch_max_train = train_set.size()[0]//batch_size
    batch_max_valid = valid_set.size()[0]//batch_size
    num_tp = train_set.size()[1]
    
    tp_hr_list = tp_hr_list.to(device)
    
    shuffle_flag = False
    
    for epoch in np.arange(max_epoch):
        
        print('Epoch ', epoch)
        
        if shuffle_flag == True:
            idx_train = torch.randperm(train_set.size()[0])
            train_set = train_set[idx_train, :, :]
            idx_valid = torch.randperm(valid_set.size()[0])
            valid_set = valid_set[idx_valid, :, :]

        train_loss = 0
        valid_loss = 0
        model.train()
        
        for b in torch.arange(batch_max_train):
            train_batch_seq = train_set[b*batch_size:(b+1)*batch_size, :, :]
            train_batch_seq = train_batch_seq.to(device)
            
            h = None ###initialize
            h_list = []
            y_list = []
            
            optimizer.zero_grad()

            for t in np.arange(num_tp - 2):
                train_batch_x = train_batch_seq[:, t, :]
                train_batch_x = torch.unsqueeze(train_batch_x, dim=1)
                train_batch_y = train_batch_seq[:, t+1, :]
                train_batch_y = torch.unsqueeze(train_batch_y, dim=1)

                h = model(train_batch_x, tp_hr_list[t].repeat(batch_size,1,1), h)

                if t == num_tp - 3:
                    h_list.append(h.permute(1,0,2))
                    y_list.append(train_batch_y)
            
            h_cat = torch.cat(h_list, dim=0)
            y_cat = torch.cat(y_list, dim=0)
            
            loss = model.loss(h_cat, y_cat)
            loss.backward()

            # loss.backward(retain_graph=True)
            optimizer.step()
            
            train_loss += loss.item()

        train_loss = train_loss/(num_tp*train_set.size()[0])            
        train_loss_list.append(train_loss)
                
        model.eval()
        with torch.no_grad():
            for b in torch.arange(batch_max_valid):
                valid_batch_seq = valid_set[b*batch_size:(b+1)*batch_size, :, :]
                valid_batch_seq = valid_batch_seq.to(device)

                h = None
                h_list = []
                y_list = []
                    
                for t in np.arange(num_tp - 2):
                    valid_batch_x = valid_batch_seq[:, t, :]
                    valid_batch_x = torch.unsqueeze(valid_batch_x, dim=1)
                    valid_batch_y = valid_batch_seq[:, t+1, :]
                    valid_batch_y = torch.unsqueeze(valid_batch_y, dim=1)

                    h = model(valid_batch_x, tp_hr_list[t].repeat(batch_size,1,1), h)

                    if t == num_tp - 3:
                        h_list.append(h.permute(1,0,2))
                        y_list.append(valid_batch_y)
            
                h_cat = torch.cat(h_list, dim=0)
                y_cat = torch.cat(y_list, dim=0)
                
                loss = model.loss(h_cat, y_cat)  
                valid_loss += loss.item()
            
        valid_loss = valid_loss/(num_tp*valid_set.size()[0])            
        valid_loss_list.append(valid_loss)
        
    return train_loss_list, valid_loss_list


def eval_dyn_model(model, test_set, batch_size, tp_hr_list, device):
    batch_max_test = test_set.size()[0]//batch_size
    num_tp = test_set.size()[1]

    test_pred_list = []
    test_targets_list = []
    test_loss = 0
    
    tp_hr_list = tp_hr_list.to(device)
    
    model.eval()
    with torch.no_grad():
        for b in torch.arange(batch_max_test):
            test_batch_seq = test_set[b*batch_size:(b+1)*batch_size, :, :]
            test_batch_seq = test_batch_seq.to(device)
            
            h = None
            h_list = []
            y_list = []
                
            for t in np.arange(num_tp - 2):
                test_batch_x = test_batch_seq[:, t, :]
                test_batch_x = torch.unsqueeze(test_batch_x, dim=1)
                test_batch_y = test_batch_seq[:, t+1, :]
                test_batch_y = torch.unsqueeze(test_batch_y, dim=1)
                
                h = model(test_batch_x, tp_hr_list[t].repeat(batch_size,1,1), h)
                h_list.append(h.permute(1,0,2))
                y_list.append(test_batch_y)
                        
            ###only want to save last time point
            t = num_tp - 2
            
            test_batch_x = test_batch_seq[:, t, :]
            test_batch_x = torch.unsqueeze(test_batch_x, dim=1)
            test_batch_y = test_batch_seq[:, t+1, :]
            test_batch_y = torch.unsqueeze(test_batch_y, dim=1)

            h = model(test_batch_x, tp_hr_list[t].repeat(batch_size,1,1), h)
            h_list.append(h.permute(1,0,2))
            y_list.append(test_batch_y)
            
            h_cat = torch.cat(h_list, dim=0)
            y_cat = torch.cat(y_list, dim=0)

            loss = model.loss(h_cat, y_cat)  
            test_loss += loss.item()
            
            test_pred_list.append(to_cpu_npy(h))
            test_targets_list.append(to_cpu_npy(test_batch_y))
    
    test_preds = np.concatenate(test_pred_list)
    test_targets = np.concatenate(test_targets_list)

    return test_loss, test_preds, test_targets


#%% Set hyperparameters

only_struct_flag = True

parser = argparse.ArgumentParser()
parser.add_argument('-stu', '--study', default='cao2019', type=str)
parser.add_argument('-cwt', '--cluster_weight_type', default='sigmoid', type=str)
parser.add_argument('-svm', '--save_vae_model_flag', default=1, type=int)
parser.add_argument('-srf', '--save_rf_flag', default=1, type=int) #50
parser.add_argument('-lvm', '--load_vae_model_flag', default=0, type=int)
parser.add_argument('-svrm', '--save_vrnn_model_flag', default=0, type=int)
parser.add_argument('-lvrm', '--load_vrnn_model_flag', default=0, type=int)
parser.add_argument('-lsdt', '--load_struct_date_time', default='2021-08-02-at-09-41-59', type=str)
parser.add_argument('-lat', '--latent_size', default=40, type=int) #1000
parser.add_argument('-mev', '--max_epoch_vae', default=250, type=int) #1000
parser.add_argument('-mer', '--max_epoch_rnn', default=10, type=int) #50
parser.add_argument('-rmf', '--return_mean_flag', default=0, type=int) #50
parser.add_argument('-bstr', '--beta_struct', nargs="+", default=[0.5, 0.5], type=float)
parser.add_argument('-bdyn', '--beta_dyn', nargs="+", default=[100, 1], type=float)


args = parser.parse_args()
study = args.study
cluster_weight_type = args.cluster_weight_type
save_vae_model_flag = args.save_vae_model_flag
load_vae_model_flag = args.load_vae_model_flag
save_vrnn_model_flag = args.save_vrnn_model_flag
load_vrnn_model_flag = args.load_vrnn_model_flag
load_struct_date_time = args.load_struct_date_time
latent_size = args.latent_size
max_epoch_vae = args.max_epoch_vae
max_epoch_rnn = args.max_epoch_rnn
return_mean_flag = args.return_mean_flag
save_rf_flag = args.save_rf_flag
beta_struct = args.beta_struct
beta_dyn = args.beta_dyn

beta_struct = torch.tensor(beta_struct, requires_grad=False)
beta_dyn = torch.tensor(beta_dyn, requires_grad=False)

batch_size = 128 #64
norm_type = 'rc' #log, rc
subsample_size = 100000 #100000
random_seed = 1
learning_rate = 1e-4

subsample_str = str(int(subsample_size/1000)) + 'k'
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, study)

today = date.today()
mdy = today.strftime("%Y-%m-%d")
clock = datetime.now()
hms = clock.strftime("%H-%M-%S")
hm = clock.strftime("%Hh-%Mm")
hm_colon = clock.strftime("%H:%M")
date_and_time = mdy + '-at-' + hms
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Load data

if study == 'cao2019':
    time_pt_list = [9.5, 10.5, 11.5, 12.5, 13.5]
    num_time_pts = len(time_pt_list)
    dense_mat_list = []
    for t in np.arange(len(time_pt_list)):
        time_pt = time_pt_list[t]
        str_time_pt = str(time_pt).replace('.', '_')
    
        fname = os.path.join(base_dir, study, 'gene_exp_mat_time_' + \
                         str_time_pt + '_' + subsample_str + '_sf_1e04_rc.mtx')
    
        sparse_mat = scipy.io.mmread(fname) #coo format
        sparse_mat = coo_matrix.transpose(sparse_mat) #rows now correspond to samples
        sparse_mat = coo_matrix.tocsr(sparse_mat) #easier to index
        dense_mat = np.array(csr_matrix.todense(sparse_mat))
        dense_mat_list.append(dense_mat)
elif study == 'schiebinger2019':
    perturbation_pt = 9
    time_pt_list = np.arange(0, perturbation_pt, 0.5)
    num_time_pts = len(time_pt_list)
    dense_mat_list = []
    for t in np.arange(len(time_pt_list)):
        time_pt = time_pt_list[t]        
        str_time_pt = str(time_pt).replace('.', '_')

        fname = os.path.join(base_dir, study, 'gene_exp_mat_time_' + \
                         str_time_pt + '_' + subsample_str + '_sf_1e04_rc.mtx')
    
        sparse_mat = scipy.io.mmread(fname) #coo format
        # sparse_mat = coo_matrix.transpose(sparse_mat) #do not transpose for schiebinger
        sparse_mat = coo_matrix.tocsr(sparse_mat) #easier to index
        dense_mat = np.array(csr_matrix.todense(sparse_mat))
        dense_mat_list.append(dense_mat)
        
        
num_cells_per_tp_list_pre = [np.size(dense_mat_list[i], 0) for i in np.arange(len(dense_mat_list))]
min_num_cells_pre = min(num_cells_per_tp_list_pre)
print('Minimum number of cells:', min_num_cells_pre, '\n')

choose_min = False
if choose_min == True:
    min_num_cells_pre = 1000
    print('Minimum number of cells:', min_num_cells_pre, '\n')

subsample_vae = True
if subsample_vae == True:
    new_dense_mat_list = []
    old_dense_mat_list = dense_mat_list
    for t in np.arange(len(time_pt_list)):
        dense_mat = dense_mat_list[t]
        cells_per_tp = np.shape(dense_mat)[0]
        
        if cells_per_tp < min_num_cells_pre:
            idcs = np.random.choice(cells_per_tp, size=min_num_cells_pre, replace=True)
        else:
            idcs = np.random.choice(cells_per_tp, size=min_num_cells_pre, replace=False)

        dense_mat = dense_mat[idcs, :]
        new_dense_mat_list.append(dense_mat)

    dense_mat_list = new_dense_mat_list

#%% From 2000 most highly variable genes, filter out genes that have no
# expression over all cells

thresh_count_min = 1 #inclusive
thresh_idcs_list = []
for d in np.arange(len(dense_mat_list)):
    dmat = dense_mat_list[d]
    d_zero_idcs = np.argwhere(np.sum(dmat, 0) <= thresh_count_min).ravel()
    thresh_idcs_list = thresh_idcs_list + d_zero_idcs.tolist()
        
thresh_idcs_list = list(set(thresh_idcs_list))
thresh_idcs_list.sort()

all_idcs_list = list(np.arange(np.shape(dense_mat)[1]))
all_idcs_set = set(all_idcs_list)
non_thresh_idcs_list = list(all_idcs_set - set(thresh_idcs_list))
non_thresh_idcs_list.sort()

dict_gene_cross_ref = {}

counter=0
for old_idx in all_idcs_list:
    if old_idx in non_thresh_idcs_list:
        dict_gene_cross_ref[old_idx] = counter
        counter+=1
    else:
        dict_gene_cross_ref[old_idx] = -1

print('Number of non-zero genes:', len(non_thresh_idcs_list), '\n')

old_dense_mat_list = dense_mat_list.copy()

thresh_count_max = np.percentile(np.concatenate(dense_mat_list, axis=0), 99.5)

print('99.5 percentile:', thresh_count_max, '\n')

dense_mat_list = []
for d in np.arange(len(old_dense_mat_list)):
    dense_mat = old_dense_mat_list[d]
    dense_mat[dense_mat > thresh_count_max] = thresh_count_max
    dense_mat_list.append(dense_mat[:,non_thresh_idcs_list])


if only_struct_flag == True:
    dense_mat_vae = np.concatenate(dense_mat_list, axis=0)
else:
    dense_mat_vae = np.concatenate(dense_mat_list[:-1], axis=0)
    
num_cells_vae, num_genes = np.shape(dense_mat_vae)

#%%
train_size, valid_size = (int(0.7*num_cells_vae), int(0.15*num_cells_vae))
test_size = num_cells_vae - train_size - valid_size

train_set, valid_set, test_set = \
    random_split(StructCustomDataset(dense_mat_vae, dense_mat_vae), \
    [train_size, valid_size, test_size])
            
train_set = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_set = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_set = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

size_determination = 'log'

if size_determination == 'user':
    latent_size = 50
    layer_size_list_struct = [num_genes, 1000, 500, latent_size]
    num_layers = int(len(layer_size_list_struct) - 1)

elif size_determination == 'user_cao':
    # latent_size = 16
    layer_size_list_struct = [num_genes, 1000, 500, latent_size]
    num_enc_layers = int(len(layer_size_list_struct) - 1)

elif size_determination == 'linear':
    num_layers = 4
    latent_size = 16
    m = -(num_genes - latent_size)/num_layers
    b = num_genes
    
    layer_size_list_struct = [num_genes]
    
    for i in np.arange(1, num_layers, 1):
        new_layer_size = int(m*i + b)
        layer_size_list_struct.append(new_layer_size)  
    layer_size_list_struct.append(latent_size)

elif size_determination == 'log':
    latent_size = 32
    num_layers = 3
    
    a = num_genes
    b = -np.log2(num_genes/latent_size)/num_layers
    
    layer_size_list_struct = [num_genes]
    
    for i in np.arange(1, num_layers, 1):
        new_layer_size = int(a*2**(b*i))
        # layer_size_list_struct_struct.append(new_layer_size)
        layer_size_list_struct.append(new_layer_size)  
    layer_size_list_struct.append(latent_size)
    
#%% Train/load structure model and output performance metrics

model = VAE_struct(num_layers, layer_size_list_struct, cluster_weight_type, beta_struct).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
print("\n"+"Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
# Load pre-trained model or train new model
if load_vae_model_flag == 1:
    load_model_filename = os.path.join(base_dir, 'cao2019', 'model_' + \
                     load_struct_date_time + '.pt')
    model.load_state_dict(torch.load(load_model_filename, \
                                 map_location=torch.device('cpu')))
    model.eval()
    model_info_filename = os.path.join(base_dir, 'cao2019', 'model_' + \
                    load_struct_date_time + '_info.txt')
    model_info_file = open(model_info_filename, "r")
    print(model_info_file.read())
    print('\n')
    model_info_file.close()
    
    vae_filename = os.path.join(base_dir, 'cao2019', 'vae-outputs-' + \
                    load_struct_date_time + '.pt')        
    with open(vae_filename, "rb") as f:
        vae_outputs_list = pickle.load(f)

    train_loss_list = vae_outputs_list[0]
    valid_loss_list = vae_outputs_list[1]
    test_loss = vae_outputs_list[2]
    test_scores = vae_outputs_list[3]
    test_labels = vae_outputs_list[4]

    max_epoch = len(train_loss_list)
    
else:
    max_epoch = max_epoch_vae
    train_loss_list, train_recon_loss_list, train_kl_loss_list, \
        valid_loss_list, valid_recon_loss_list, valid_kl_loss_list = \
        train_struct_model(model, optimizer, train_set, valid_set, max_epoch, device)

    test_loss, test_scores, test_labels, \
        k_mu, k_logvar, k_weights, k = eval_struct_model(model, test_set, device)


mean_gene_exp_pred = np.mean(test_scores, axis=0)
mean_gene_exp_labels = np.mean(test_labels, axis=0)
test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)

elapsed = (time.time() - start_time)
elapsed_h = int(elapsed//3600)
elapsed_m = int((elapsed - elapsed_h*3600)//60)
elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
print('Elapsed time for training: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))

print('Test PCC:', test_pcc)
print('Test SCC:', test_scc)


# #%% With structure model trained, transform input space coordinates to latent space coordinates

# model.eval()
# k_mu_mat_list = []
# k_logvar_mat_list = []
# k_weights_mat_list = []

# for t in np.arange(len(dense_mat_list)):
#     dense_mat = dense_mat_list[t]
#     dense_mat_struct_input = StructCustomDataset(dense_mat, dense_mat)

#     x_all = DataLoader(dataset=dense_mat_struct_input, batch_size=batch_size, shuffle=False)
    
#     k_mu_list = []
#     k_logvar_list = []
#     k_weights_list = []

#     with torch.no_grad():
#         for x_batch, _ in x_all:
#             x_batch = x_batch.to(device)

#             forward_scores, k_mu, k_logvar, k_weights, k = model(x_batch)
            
#             k_mu_list.append(k_mu)
#             k_logvar_list.append(k_logvar)
#             k_weights_list.append(k_weights)
        
#     k_mu_mat = torch.cat(k_mu_list, dim=0)
#     k_logvar_mat = torch.cat(k_logvar_list, dim=0)
#     k_weights_mat =  torch.cat(k_weights_list, dim=0)
    
#     k_mu_mat_list.append(k_mu_mat)
#     k_logvar_mat_list.append(k_logvar_mat)
#     k_weights_mat_list.append(k_weights_mat)


# num_cells_per_tp_list = [np.size(k_mu_mat_list[i], 0) for i in np.arange(len(k_mu_mat_list))]

# num_test_cells = num_cells_per_tp_list[-1]
# # num_test_cells = 10000

# if num_test_cells > min(num_cells_per_tp_list):
#     replace_flag = True
# else:
#     replace_flag = False

# k_mat_sampled_list = []
# dense_mat_sampled_list = []
# for t in np.arange(num_time_pts):
#     s = num_cells_per_tp_list[t]
    
#     k_mu_mat = k_mu_mat_list[t]
#     k_logvar_mat = k_logvar_mat_list[t]
#     k_weights_mat = k_weights_mat_list[t]
    
#     k_mat = torch.cat([k_mu_mat, k_logvar_mat, k_weights_mat], dim=1)
    
#     dense_mat = torch.from_numpy(dense_mat_list[t])
        
#     idcs = np.random.choice(s, size = num_test_cells, replace=replace_flag)
#     k_mat_sampled = torch.unsqueeze(k_mat[idcs, :], 1)
#     k_mat_sampled_list.append(k_mat_sampled)
    
#     dense_mat_sampled = torch.unsqueeze(dense_mat[idcs, :], 1)
#     dense_mat_sampled_list.append(dense_mat_sampled)
    
# k_mat_sampled_cat = torch.cat(k_mat_sampled_list, 1)
# ### num_cells x timepoint x latent_dim
# dense_mat_sampled_cat = torch.cat(dense_mat_sampled_list, 1)

# train_size = int(0.7*num_test_cells)
# valid_size = num_test_cells - train_size

# train_set_list = []
# valid_set_list = []
# test_set_list = []

# ### for random permute
# idx = torch.randperm(num_test_cells)
# k_mat_sampled_cat = k_mat_sampled_cat[idx, :, :]
# dense_mat_sampled_cat = dense_mat_sampled_cat[idx, :, :]

# train_set = k_mat_sampled_cat[:train_size, :-1, :]
# valid_set = k_mat_sampled_cat[train_size:, :-1, :]
# test_set = k_mat_sampled_cat

# train_set_rc = dense_mat_sampled_cat[:train_size, :-1, :]
# valid_set_rc = dense_mat_sampled_cat[train_size:, :-1, :]
# test_set_rc = dense_mat_sampled_cat



# #%% Match cells at time point t with cells at time point t + 1

# ###https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

# new_start_time = time.time()

# sort_type = 'lsa_ent'

# np_train_set_kmat = to_cpu_npy(train_set)
# np_valid_set_kmat = to_cpu_npy(valid_set)
# np_test_set_kmat = to_cpu_npy(test_set)

# num_cells_train = np.size(np_train_set_kmat, 0)
# num_cells_valid = np.size(np_valid_set_kmat, 0)
# num_tp_except_last = np.size(np_train_set_kmat, 1)
# num_tp = np.size(np_test_set_kmat, 1)

# shuffle_flag = False

# alpha = 0.5

# for t in np.arange(num_tp - 2):
    
#     print('t: ' + str(t) + '\n')
    
#     pd_test = calc_dist(np_test_set_kmat[:, t, :], np_test_set_kmat[:, t+1, :], \
#                            latent_size)
#     first_tp_ind, second_tp_ind = linear_sum_assignment(pd_test, maximize=True)

#     new_slice_test_kmat = np_test_set_kmat[second_tp_ind, t+1, :]
#     np_test_set_kmat[:, t+1, :] = new_slice_test_kmat
    
#     if t < num_tp - 2:
        
#         pd_train = calc_dist(np_train_set_kmat[:, t, :], np_train_set_kmat[:, t+1, :], \
#                                latent_size, alpha)        
#         first_tp_ind, second_tp_ind = linear_sum_assignment(pd_train, maximize=True)
        
#         new_slice_train_kmat = np_train_set_kmat[second_tp_ind, t+1, :]
#         np_train_set_kmat[:, t+1, :] = new_slice_train_kmat
        
        
#         pd_valid = calc_dist(np_valid_set_kmat[:, t, :], np_valid_set_kmat[:, t+1, :], \
#                                latent_size, alpha)
#         first_tp_ind, second_tp_ind = linear_sum_assignment(pd_valid, maximize=True)
        
#         new_slice_valid_kmat = np_valid_set_kmat[second_tp_ind, t+1, :]
#         np_valid_set_kmat[:, t+1, :] = new_slice_valid_kmat
        
#     elapsed_min = np.round((time.time() - new_start_time)/60, 2)
#     print('elapsed minutes cumulative: ' + str(elapsed_min) + ' min\n')

# train_set = torch.from_numpy(np_train_set_kmat)
# valid_set = torch.from_numpy(np_valid_set_kmat)
# test_set = torch.from_numpy(np_test_set_kmat)


# #%% run dynamics model

# max_epoch = max_epoch_rnn
# keypoint_size = hidden_size = latent_size
# loss_type = 'MSE_KL_weighted'   #'MSE', 'MSE_weighted', KL', 'KL_weighted', \
#     # 'wass', 'wass_weighted', 'MSE_KL_weighted', 'Wass_KL_weighted'

# test_pcc_list = []
# test_scc_list = []

# ### to double check param values: model.decoder_module.decoders[0].weight.data
# struct_model = model
# for param in struct_model.parameters():
#     param.requires_grad = False

# # tp_hr_list = torch.tensor([9.5, 10.5, 11.5, 12.5, 13.5])
# # tp_hr_list = (tp_hr_list - tp_hr_list[0].repeat(5))
# tp_hr_list = torch.tensor([0]).repeat(5)
    
# for i in np.arange(20):
#     print('\nIteration: ' + str(i) + '\n')
    
#     dyn_model = VRNN(batch_size, hidden_size, keypoint_size, loss_type).to(device)
#     dyn_optimizer = torch.optim.Adam(dyn_model.parameters(), lr = learning_rate)
    
#     # print("\n"+"Model's state_dict:")
#     # for param_tensor in dyn_model.state_dict():
#     #     print(param_tensor, "\t", dyn_model.state_dict()[param_tensor].size())
        
#     train_loss_list, valid_loss_list = \
#         train_dyn_model(dyn_model, dyn_optimizer, train_set, valid_set, max_epoch, batch_size, tp_hr_list, device)
#     test_loss, test_pred, test_targets = eval_dyn_model(dyn_model, test_set, batch_size, tp_hr_list, device)
    
#     test_pred = np.reshape(test_pred, (-1, keypoint_size*3))
#     test_pred_tensor = torch.from_numpy(test_pred)
#     model.eval()
#     with torch.no_grad():
#         y_pred = model.forward_predict(test_pred_tensor, hidden_size)
    
#     y_pred = y_pred.numpy()
#     y_true = test_set_rc[:, -1, :].numpy()
#     batch_size_effect_dim = min(np.shape(y_true)[0], np.shape(y_pred)[0])
    
#     y_true = y_true[:batch_size_effect_dim, :]
#     y_pred = y_pred[:batch_size_effect_dim, :]
    
#     mean_gene_exp_pred = np.mean(y_pred, axis=0)
#     mean_gene_exp_labels = np.mean(y_true, axis=0)
#     test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
#     test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
#     test_pcc_list.append(test_pcc)
#     test_scc_list.append(test_scc)

# elapsed = (time.time() - new_start_time)
# elapsed_h = int(elapsed//3600)
# elapsed_m = int((elapsed - elapsed_h*3600)//60)
# elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
# print('\nElapsed cumulative time: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))


# print('\nBaseline performance from previous time point:')
# mean_gene_exp_previous = np.mean(dense_mat_list[-2][:batch_size_effect_dim, :], axis=0)
# baseline_pcc, _ = pearsonr(mean_gene_exp_previous, mean_gene_exp_labels)
# baseline_scc, _ = spearmanr(mean_gene_exp_previous, mean_gene_exp_labels)
# print('Baseline PCC:', baseline_pcc)
# print('Baseline SCC:', baseline_scc)


# elapsed = (time.time() - new_start_time)
# elapsed_h = int(elapsed//3600)
# elapsed_m = int((elapsed - elapsed_h*3600)//60)
# elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
# print('Total elapsed time for training: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))



# #%% Plot figures

# plot_figures = True

# if plot_figures == True:
#     start_time = time.time()
    
#     test_pcc_str = str(np.round(test_pcc, 3))
#     test_scc_str = str(np.round(test_scc, 3))
#     cc_str = (f"Test PCC = {test_pcc_str}\n"
#               f"Test SCC = {test_scc_str}")
    
#     y_prev = dense_mat_list[-2][:batch_size_effect_dim, :]
#     scores_and_labels = np.concatenate((y_true, y_pred, y_prev), axis=0)    
#     trans = umap.UMAP(random_state=0).fit(scores_and_labels)
    
#     epochs = np.arange(0, max_epoch, 1)
    
    
#     if cluster_weight_type == 'softmax':
#         cluster_weight_type_str = 'Softmax Latent Weights'
#     elif cluster_weight_type == 'sigmoid':
#         cluster_weight_type_str = 'Sigmoidal Latent Weights'
#     else:
#         cluster_weight_type_str = 'No Weights'
        
#     if loss_type == 'KL':
#         loss_type_str = 'KL Dynamics Loss'
#     elif loss_type == 'KL_weighted':
#         loss_type_str = 'Weighted KL Dynamics Loss'
#     elif loss_type == 'wass':
#         loss_type_str = 'Wasserstein Dynamics Loss'
#     elif loss_type == 'wass_weighted':
#         loss_type_str = 'Weighted Wasserstein Dynamics Loss'
#     elif loss_type == 'MSE_KL_weighted':
#         loss_type_str = 'MSE KL weighted'
#     elif loss_type == 'Wass_KL_weighted':
#         loss_type_str = 'Wasserstein KL weighted'
        
#     mpl.rcdefaults()
#     # plt.rcParams.update({'figure.autolayout': True})
#     # plt.style.use('seaborn')
    
    
#     fig = plt.figure(figsize = (12, 5))
#     plt.suptitle('Test Predictions with Default Model', fontsize=16, y=0.95)
#     fig.subplots_adjust(top=0.8)
#     subtitle_fontsize = 14
#     axes_labels_fontsize = 12
    
#     ax1 = fig.add_subplot(121)
#     ax1.plot(epochs, train_loss_list, 'orange', label='Training')
#     ax1.plot(epochs, valid_loss_list, 'blue', label='Validation')
#     ax1.set_xlabel('Epochs', fontsize = axes_labels_fontsize)
#     ax1.set_ylabel('Loss', fontsize = axes_labels_fontsize)
#     ax1.set_title('Loss Curves', fontsize = subtitle_fontsize)
#     ax1.legend()
#     ax1.text(0.1, 0.1, cc_str, horizontalalignment='left', verticalalignment='center', \
#          transform=ax1.transAxes)
    
    
#     ax2 = fig.add_subplot(122)
#     num_cells_test = np.size(y_true, 0)
#     ax2.scatter(trans.embedding_[:num_cells_test, 0], trans.embedding_[:num_cells_test:, 1], color='blue', alpha=0.5, s=2, label='True')
#     ax2.scatter(trans.embedding_[num_cells_test:2*num_cells_test, 0], trans.embedding_[num_cells_test:2*num_cells_test, 1], color='orange', alpha=0.75, s=2, label='Simulated')
#     ax2.scatter(trans.embedding_[2*num_cells_test:, 0], trans.embedding_[2*num_cells_test:, 1], color='green', alpha=0.1, s=2, label='Baseline')
    
    
#     ax2.legend()
#     ax2.set_title('UMAP', fontsize = subtitle_fontsize)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_xlabel('Dimension 1', fontsize = axes_labels_fontsize)
#     ax2.set_ylabel('Dimension 2', fontsize = axes_labels_fontsize)
    
#     mpl.rcdefaults()
    
#     elapsed = (time.time() - start_time)
#     elapsed_h = int(elapsed//3600)
#     elapsed_m = int((elapsed - elapsed_h*3600)//60)
#     elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
#     print('Elapsed time for training and UMAP: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))


# #%% Save random forest outputs, VAE model, VRNN model

    
if save_rf_flag == 1:
    random_forest_data = [test_scores, test_labels]
    random_forest_data_file = os.path.join(base_dir, study, 'rf-inputs-' + \
        load_struct_date_time + '.pt')
    with open(random_forest_data_file, 'wb') as handle:
        pickle.dump(random_forest_data, handle)

if save_vae_model_flag == 1:
    save_model_filename = os.path.join(base_dir, study, 'model_' + \
                      date_and_time + '.pt')
    torch.save(model.state_dict(), save_model_filename)
    
    model_info_filename = os.path.join(base_dir, study, 'model_' + \
                    date_and_time + '_info.txt')
    f = open(model_info_filename, 'w')
    # f.write('File name: ' + os.path.join(base_dir, 'vae_rnn_detached_v2.py' + '\n'))

    f.write('File name: ' + os.path.basename(__file__) + '\n')
    f.write('Study: ' + study  + '\n')
    f.write('Model reference date and time: ' + date_and_time + '\n\n')
    f.write('Model class: Default\n')
    f.write('Start date: ' + mdy + '\n')
    f.write('Start time: ' + hm_colon + '\n')
    f.write('Total time: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))
    f.write('\n\n')
    f.write('Hyperparameters:\n')
    f.write('Return mean flag: ' + str(return_mean_flag) + '\n')
    f.write('Max epoch VAE:' + str(max_epoch_vae) + '\n')
    f.write('Beta struct: ' + str(beta_struct[0]) + ', ' + str(beta_struct[1]) + '\n')
    f.write('Cluster weight type: ' + cluster_weight_type + '\n')
    f.write('Batch size: ' + str(batch_size) + '\n')
    f.write('Subsample size: ' + str(subsample_size) + '\n')
    f.write('Learning rate: ' + str(learning_rate) + '\n')
    f.write('Input data norm type: ' + norm_type + '\n\n')
    f.write('Performance:\n')
    f.write('PCC:' + str(test_pcc) + '\n')
    f.write('SCC:' + str(test_scc) + '\n\n')
    f.write('Model\'s state_dict:\n')
    for param_tensor in model.state_dict():
        f.write(str(param_tensor) + "\t" + str(model.state_dict()[param_tensor].size()) + '\n')
    f.close()
    
    vae_outputs_list = [train_loss_list, train_recon_loss_list, train_kl_loss_list, \
        valid_loss_list, valid_recon_loss_list, valid_kl_loss_list, \
        test_loss, test_scores, test_labels]
    vae_outputs_filename = os.path.join(save_dir, 'vae-outputs-' \
                                            + date_and_time + '.pt')
    with open(vae_outputs_filename, 'wb') as handle:
        pickle.dump(vae_outputs_list, handle)
    
    
    gene_cross_ref_dict_filename = os.path.join(save_dir, 'gene-crf-dict-' \
                                            + date_and_time + '.pt')
    with open(gene_cross_ref_dict_filename, 'wb') as handle:
        pickle.dump(dict_gene_cross_ref, handle)

# if save_vrnn_model_flag == 1:
#     save_model_filename = os.path.join(base_dir, 'cao2019', 'model_rnn_' + \
#                      date_and_time + '.pt')
#     torch.save(dyn_model.state_dict(), save_model_filename)
#     model_info_filename = os.path.join(base_dir, 'cao2019', 'model_rnn_' + \
#                     date_and_time + '_info.txt')
#     f = open(model_info_filename, 'w')
#     f.write('File name: ' + os.path.join(base_dir, 'vrnn_exp_thres_zprob_v1.py' + '\n'))

#     # f.write('File name: ' + os.path.basename(__file__) + '\n')
#     f.write('Model reference date and time: ' + date_and_time + '\n\n')
#     f.write('Start date: ' + mdy + '\n')
#     f.write('Start time: ' + hm_colon + '\n')
#     f.write('Total time: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))
#     f.write('\n\n')
#     f.write('Model architecture type: tanh/exp/thresh\n\n')
#     f.write('Hyperparameters:\n')
#     f.write('Max epoch RNN:' + str(max_epoch_rnn) + '\n')
#     f.write('Adversarial: No\n')
#     f.write('Beta dyn: ' + str(beta_dyn[0]) + ', ' + str(beta_dyn[1]) + '\n')
#     f.write('Cluster weight type: ' + cluster_weight_type + '\n')
#     f.write('Cluster weight type: ' + cluster_weight_type + '\n')
#     f.write('Max epoch: ' + str(max_epoch) + '\n')
#     f.write('Batch size: ' + str(batch_size) + '\n')
#     f.write('Subsample size: ' + str(subsample_size) + '\n')
#     f.write('Learning rate: ' + str(learning_rate) + '\n')
#     f.write('Input data norm type: ' + norm_type + '\n\n')
#     f.write('Model\'s state_dict:\n')
#     for param_tensor in dyn_model.state_dict():
#         f.write(str(param_tensor) + "\t" + str(dyn_model.state_dict()[param_tensor].size()) + '\n')
#     f.close()

#     vrnn_outputs_list = [train_loss_list, valid_loss_list, test_loss, test_pred, test_targets]
#     vrnn_outputs_list_file = os.path.join(save_dir, 'vrnn-outputs-' \
#                                             + date_and_time + '.pt')
#     with open(vrnn_outputs_list_file, 'wb') as handle:
#         pickle.dump(vrnn_outputs_list, handle)
        
        
# #%% More figures 

# plot_figures = False
# if plot_figures == True:
#     scores_and_labels = np.concatenate((test_labels, test_scores), axis=0)
#     trans = umap.UMAP(random_state=0).fit(scores_and_labels)
    
#     max_epoch = np.size(train_loss_list, 0)
#     epochs = np.arange(0, max_epoch, 1)
#     test_pcc_str = str(np.round(test_pcc, 3))
#     test_scc_str = str(np.round(test_scc, 3))
#     cc_str = (f"Test PCC = {test_pcc_str}\n"
#               f"Test SCC = {test_scc_str}")
#     test_scores_vec = test_scores.flatten()
#     test_labels_vec = test_labels.flatten()
    
    
#     mpl.rcdefaults()
#     plt.rcParams.update({'figure.autolayout': True})
#     path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
#     prop = font_manager.FontProperties(fname=path)
#     mpl.rcParams['font.family'] = prop.get_name()
#     fontname = mpl.rcParams['font.family']
    
        
#     fig = plt.figure(dpi=300)
#     suptitle_fontsize = 14
#     subtitle_fontsize = 12
#     axes_labels_fontsize = 10
#     axes_tick_labels_fontsize = 10
#     legend_fontsize = 8
    
#     plt.suptitle('ET Model Performance', \
#                   y=0.95, fontsize=suptitle_fontsize)
#     fig.subplots_adjust(top=0.8)
    
    
#     ax1 = fig.add_subplot(121)
#     ax1.plot(epochs, train_loss_list, 'orange', label='Training')
#     ax1.plot(epochs, valid_loss_list, 'blue', label='Validation')
    
#     ax1.set_xlabel('Epochs', fontsize = axes_labels_fontsize)
#     ax1.set_ylabel('Loss', fontsize = axes_labels_fontsize)
#     ax1.set_title('Model Loss', fontsize = subtitle_fontsize)
#     ax1.legend(fontsize = legend_fontsize)
#     ax1.text(0.1, 0.1, cc_str, horizontalalignment='left', verticalalignment='center', \
#          transform=ax1.transAxes)
#     ax1.tick_params(axis='x', labelsize = axes_tick_labels_fontsize)
#     ax1.tick_params(axis='y', labelsize = axes_tick_labels_fontsize)
    
#     ax2 = fig.add_subplot(122)
#     num_cells_test = np.size(test_labels, 0)
#     ax2.scatter(trans.embedding_[num_cells_test:, 0], trans.embedding_[num_cells_test:, 1], color='blue', alpha=0.5, s=2, label='True')
#     ax2.scatter(trans.embedding_[:num_cells_test, 0], trans.embedding_[:num_cells_test, 1], color='orange', alpha=0.5, s=2, label='Simulated')
#     # ax2.scatter(embedding_labels[:, 0], embedding_labels[:, 1], color='blue', alpha=0.5, s=2, label='True')
#     # ax2.scatter(embedding_scores[:, 0], embedding_scores[:, 1], color='orange', alpha=0.5, s=2, label='Simulated')
    
#     ax2.legend(fontsize = legend_fontsize)
#     ax2.set_title('UMAP', fontsize = subtitle_fontsize)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_xlabel('Dimension 1', fontsize = axes_labels_fontsize)
#     ax2.set_ylabel('Dimension 2', fontsize = axes_labels_fontsize)
#     ax2.tick_params(axis='x', labelsize = axes_tick_labels_fontsize)
#     ax2.tick_params(axis='y', labelsize = axes_tick_labels_fontsize)
    
    
#     scores_mat = np.vstack((test_scores_vec, test_labels_vec)).T
    
#     xmin_hist = min(np.min(test_scores_vec), np.min(test_labels_vec))
#     xmax_hist = max(np.max(test_scores_vec), np.max(test_labels_vec))
#     xmin_hist_disp = np.floor(xmin_hist/100)*100
#     xmax_hist_disp = np.ceil(xmax_hist/100)*100
#     hist_bins = np.arange(xmin_hist_disp, xmax_hist_disp+100, 100)
    
#     xmin_hist_trunc = min(np.percentile(test_scores_vec, 1), np.percentile(test_labels_vec, 1))
#     xmax_hist_trunc = max(np.percentile(test_scores_vec, 99), np.percentile(test_labels_vec, 99))
#     xmin_hist_trunc_disp = np.floor(xmin_hist_trunc/10)*10
#     xmax_hist_trunc_disp = np.ceil(xmax_hist_trunc/10)*10
#     hist_bins_trunc = np.arange(xmin_hist_trunc_disp, xmax_hist_trunc_disp+10, 10)
    
#     num_gene_exp_vals = np.size(test_scores_vec)
#     hist_test_scores, _ = np.histogram(test_scores_vec, bins=hist_bins)
#     hist_test_labels, _ = np.histogram(test_labels_vec, bins=hist_bins)
#     hist_test_scores_trunc, _ = np.histogram(test_scores_vec, bins=hist_bins_trunc)
#     hist_test_labels_trunc, _ = np.histogram(test_labels_vec, bins=hist_bins_trunc)
    
#     print('Predicted:')
#     print('Minimum: ', np.round(np.min(test_scores_vec), 2))
#     print('Q1: ', np.round(np.percentile(test_scores_vec, 25), 2))
#     print('Q3: ', np.round(np.percentile(test_scores_vec, 75), 2))
#     print('Maximum: ', np.round(np.max(test_scores_vec), 2), '\n')
    
#     print('True:')
#     print('Minimum: ', np.round(np.min(test_labels_vec), 2))
#     print('Q1: ', np.round(np.percentile(test_labels_vec, 25), 2))
#     print('Q3: ', np.round(np.percentile(test_labels_vec, 75), 2))
#     print('Maximum: ', np.round(np.max(test_labels_vec), 2), '\n')
    
    
#     # fig = plt.figure(figsize = (15, 4))
#     fig = plt.figure(dpi=300)
#     plt.rcParams.update({'figure.autolayout': True})
#     path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
#     prop = font_manager.FontProperties(fname=path)
#     mpl.rcParams['font.family'] = prop.get_name()
#     fontname = mpl.rcParams['font.family']
    
#     suptitle_fontsize = 14
#     subtitle_fontsize = 12
#     axes_labels_fontsize = 10
#     axes_tick_labels_fontsize = 10
#     legend_fontsize = 8
    
    
#     plt.suptitle('ET Model: Truncated Histograms of Gene Expression Values', \
#                   fontsize=12, y=0.95)
    
#     fig.subplots_adjust(top=0.8)
#     subtitle_fontsize = 10
#     axes_labels_fontsize = 8
#     axes_tick_labels_fontsize = 8
#     legend_fontsize = 6
    
#     ax2 = fig.add_subplot(121)
#     ax2.set_title('Linear Scale', fontsize = subtitle_fontsize)
#     xmin_ecdf = 0
#     xmax_ecdf = max(np.percentile(test_scores_vec, 95), np.percentile(test_labels_vec, 95))
#     xrange = np.linspace(xmin_ecdf-1, xmax_ecdf+1, 50)
    
#     bool_vec_scores = np.reshape(test_scores_vec < xmin_ecdf, (-1, 1))
#     test_scores_vec_trunc = np.reshape(np.copy(test_scores_vec), (-1, 1))
#     test_scores_vec_trunc[bool_vec_scores] = xmin_ecdf
    
#     bool_vec_labels = np.reshape(test_labels_vec < xmin_ecdf, (-1, 1))
#     test_labels_vec_trunc = np.reshape(np.copy(test_labels_vec), (-1, 1))
#     test_labels_vec_trunc[bool_vec_labels] = xmin_ecdf 
    
#     ax2.hist(test_labels_vec_trunc, bins=xrange, \
#              label='True Distribution of Values', color='blue', \
#                  alpha=0.5)
#     ax2.hist(test_scores_vec_trunc, bins=xrange, \
#               label='Predicted Distribution of Values', color='orange', \
#                   alpha=0.5)
    
#     ax2.set_xlim([xmin_ecdf, xmax_ecdf])
#     ax2.set_xlabel('Gene Expression Values', fontsize= axes_labels_fontsize)
#     ax2.set_ylabel('Frequency', fontsize= axes_labels_fontsize)
#     ax2.tick_params(axis='x', labelsize= axes_tick_labels_fontsize) #, rotation=45)
#     ax2.tick_params(axis='y', labelsize= axes_tick_labels_fontsize)
#     ax2.legend(fontsize = legend_fontsize)
    
    
#     ax2 = fig.add_subplot(122)
#     ax2.set_title('Log Scale', fontsize = subtitle_fontsize)
#     xmin_ecdf = 1/10
#     xmax_ecdf = max(np.max(test_scores_vec), np.max(test_labels_vec))
#     xrange = np.logspace(np.log10(xmin_ecdf), np.log10(xmax_ecdf), 50)
    
#     bool_vec_scores = np.reshape(test_scores_vec < xmin_ecdf, (-1, 1))
#     test_scores_vec_trunc = np.reshape(np.copy(test_scores_vec), (-1, 1))
#     test_scores_vec_trunc[bool_vec_scores] = xmin_ecdf
    
#     bool_vec_labels = np.reshape(test_labels_vec < xmin_ecdf, (-1, 1))
#     test_labels_vec_trunc = np.reshape(np.copy(test_labels_vec), (-1, 1))
#     test_labels_vec_trunc[bool_vec_labels] = xmin_ecdf 
    
#     ax2.hist(test_labels_vec_trunc, bins=xrange, \
#              label='True Distribution of Values', color='blue', \
#                  alpha=0.5, log=True)
#     ax2.hist(test_scores_vec_trunc, bins=xrange, \
#               label='Predicted Distribution of Values', color='orange', \
#                   alpha=0.5, log=True)
    
#     ax2.set_xlim([xmin_ecdf, xmax_ecdf])
#     ax2.set_xscale('log')
#     ax2.set_xlabel('Gene Expression Values', fontsize= axes_labels_fontsize)
#     ax2.set_ylabel('Frequency', fontsize= axes_labels_fontsize)
#     ax2.tick_params(axis='x', labelsize= axes_tick_labels_fontsize) #, rotation=45)
#     ax2.tick_params(axis='y', labelsize= axes_tick_labels_fontsize)
#     ax2.legend(fontsize = legend_fontsize)
    
    
#     fig = plt.figure(dpi=300, figsize=(8,3))
#     plt.rcParams.update({'figure.autolayout': True})
#     path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
#     prop = font_manager.FontProperties(fname=path)
#     mpl.rcParams['font.family'] = prop.get_name()
#     fontname = mpl.rcParams['font.family']
    
#     suptitle_fontsize = 12
#     subtitle_fontsize = 10
#     axes_labels_fontsize = 8
#     axes_tick_labels_fontsize = 8
#     legend_fontsize = 6
    
#     ax1 = fig.add_subplot(121)
#     ax1.set_title('ECDF for Gene Expression Values', fontsize = subtitle_fontsize)
#     ecdf_y_pred = ECDF(test_scores_vec)
#     ecdf_y_true = ECDF(test_labels_vec)
#     xmin_ecdf = 1e-2
#     xmax_ecdf = max(np.max(test_scores_vec), np.max(test_labels_vec))
#     xrange = np.logspace(np.log10(xmin_ecdf), np.log10(xmax_ecdf), 50)
#     # xrange = np.linspace(xmin_ecdf, xmax_ecdf, 100)
    
#     ax1.plot(xrange, ecdf_y_true(xrange), label='True Distribution of Values', color='blue')
#     ax1.plot(xrange, ecdf_y_pred(xrange), label='Predicted Distribution of Values', color='orange')
#     ax1.legend(fontsize = legend_fontsize)
#     ax1.set_xscale('log')
#     ax1.set_xlabel('Gene Expression Values', fontsize= axes_labels_fontsize)
#     ax1.set_ylabel('Cumulative Probability', fontsize= axes_labels_fontsize)
#     ax1.set_xlim([xmin_ecdf, xmax_ecdf])
#     ax1.set_ylim([0.7, 1.04])
#     ax1.tick_params(axis='x', labelsize= axes_tick_labels_fontsize)
#     ax1.tick_params(axis='y', labelsize= axes_tick_labels_fontsize)
    
#     ax2 = fig.add_subplot(122)
#     ax2.set_title('ECDF for Mean Gene Expression', fontsize = subtitle_fontsize)
#     ecdf_y_pred = ECDF(np.mean(test_scores, axis=0))
#     ecdf_y_true = ECDF(np.mean(test_labels, axis=0))
#     xmin_ecdf = 1e-4
#     xmax_ecdf = max(np.max(np.mean(test_scores, axis=0)), np.max(np.mean(test_labels, axis=0)))
#     xrange = np.logspace(np.log10(xmin_ecdf), np.log10(xmax_ecdf), 50)
#     # xrange = np.linspace(xmin, xmax, 100)
#     ax2.plot(xrange, ecdf_y_true(xrange), label='True Distribution of Means', color='blue')
#     ax2.plot(xrange, ecdf_y_pred(xrange), label='Predicted Distribution of Means', color='orange')
#     ax2.legend(fontsize = legend_fontsize)
#     ax2.set_xscale('log')
#     ax2.set_xlabel('Gene Expression Means', fontsize= axes_labels_fontsize)
#     ax2.set_ylabel('Cumulative Probability', fontsize= axes_labels_fontsize)
#     ax2.set_ylim([0, 1.04])
#     ax2.tick_params(axis='x', labelsize= axes_tick_labels_fontsize)
#     ax2.tick_params(axis='y', labelsize= axes_tick_labels_fontsize)
    
#     fig.subplots_adjust(bottom=0.2)
    
#     mpl.rcdefaults()




    
