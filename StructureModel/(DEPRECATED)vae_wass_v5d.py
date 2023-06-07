import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy as scp
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr

import argparse
import random
import time
from datetime import datetime, date

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, random_split, DataLoader, Subset

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.stats import linregress
from scipy.stats import gaussian_kde


import umap
from geomloss import SamplesLoss


def to_cpu_npy(x):
    if type(x) == list:
        new_x = []
        for element in x:
            new_x.append(element.detach().cpu().numpy())
    else:
        new_x = x.detach().cpu().numpy()
    return new_x


class CustomDataset(Dataset):
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
    def __init__(self, num_layers, layer_size_list, cluster_weight_type):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList()

        for i in np.arange(0, num_layers - 1, 1):
            self.encoders.append(nn.Linear(layer_size_list[i], layer_size_list[i+1]))

        self.z_mu_transform = nn.Linear(layer_size_list[-2], layer_size_list[-1])
        self.z_logvar_transform = nn.Linear(layer_size_list[-2], layer_size_list[-1])
        self.z_cluster_weight_transform = nn.Linear(layer_size_list[-2], layer_size_list[-1])

        self.softmax_func = nn.Softmax(dim=1)
        self.cluster_weight_type = cluster_weight_type

    def forward(self, x):
        z = x

        for i in range(len(self.encoders)):

            z = self.encoders[i](z)
            z = torch.relu(z)

        z_mu = self.z_mu_transform(z)       
        z_logvar = self.z_logvar_transform(z)      
        
        if self.cluster_weight_type != 'vanilla':
            z_weights = self.z_cluster_weight_transform(z)
            if self.cluster_weight_type == 'softmax':
                z_weights = self.softmax_func(z_weights)
            elif self.cluster_weight_type == 'sigmoid':
                z_weights = torch.sigmoid(z_weights)

        else:
            z_weights = torch.empty((0))
            
        return z_mu, z_logvar, z_weights


class Decoder(nn.Module):
    def __init__(self, num_layers, layer_size_list):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList()
        for i in np.arange(num_layers, 0, -1):
            self.decoders.append(nn.Linear(layer_size_list[i], layer_size_list[i-1]))
            
    def forward(self, sample):
        y = sample
        
        for i in range(len(self.decoders) - 1):      
            y = self.decoders[i](y)
            y = torch.relu(y)

        y = self.decoders[-1](y)
        
        
        return y
    
    
class VAE(nn.Module):
    def __init__(self, num_layers, layer_size_list, cluster_weight_type='vanilla', beta=0.5):
        super(VAE, self).__init__()
        self.encoder_module = Encoder(num_layers, layer_size_list, cluster_weight_type)
        self.decoder_module = Decoder(num_layers, layer_size_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cluster_weight_type = cluster_weight_type
        self.beta = beta

    def forward(self, x):
        z_mu, z_logvar, z_weights = self.encoder_module(x)
        sample = self.latent_sample(z_mu, z_logvar, z_weights)
        y = self.decoder_module(sample)
        
        return y, z_mu, z_logvar
    
    def loss(self, x, y, z_mu, z_logvar):
        
        recon_loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        recon_loss = recon_loss_func(x, y)

        KL_loss = -0.5*torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        vae_loss = 2*((1-self.beta)*recon_loss + self.beta*KL_loss)

        return vae_loss

    def latent_sample(self, z_mu, z_logvar, z_weights):

        if self.training:
            # the reparameterization trick

            std = z_logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            sample = eps.mul(std).add_(z_mu)

        else:
            sample = z_mu
        
        if self.cluster_weight_type != 'vanilla':     
            ###element-wise multiplication
            sample = torch.mul(sample, z_weights)
        
        return sample
    
        
def train_model(model, optimizer, train_set, valid_set, max_epoch, device):

    train_loss_list = []
    valid_loss_list = []
    
    for epoch in np.arange(max_epoch):
        
        print('Epoch ', epoch)

        train_loss = 0
        valid_loss = 0

        model.train()
        train_pred_list = []        
        train_labels_list = []
        
        for x_batch, _ in train_set:
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            forward_scores, z_mu, z_logvar = model(x_batch)

            loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()

            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            train_pred_list.append(to_cpu_npy(pred))
            train_labels_list.append(to_cpu_npy(x_batch))
            
        train_loss_list.append(train_loss/len(train_labels_list))
        
        model.eval()
        valid_pred_list = []        
        valid_labels_list = []
        with torch.no_grad():
            # for x_batch in valid_set:
            for x_batch, _ in valid_set:
                x_batch = x_batch.to(device)
                
                forward_scores, z_mu, z_logvar = model(x_batch)
                loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
                valid_loss += loss.item()

                pred = torch.reshape(forward_scores, (len(forward_scores), -1))
                valid_pred_list.append(to_cpu_npy(pred))
                valid_labels_list.append(to_cpu_npy(x_batch))

        valid_loss_list.append(valid_loss/len(valid_labels_list))

    return train_loss_list, valid_loss_list


def eval_model(model, test_set, device):
    model.eval()
    test_loss = 0
    test_pred_list = []
    test_labels_list = []

    with torch.no_grad():
        # for x_batch in test_set:
        for x_batch, _ in test_set:
            x_batch = x_batch.to(device)
                
            forward_scores, z_mu, z_logvar = model(x_batch)
            loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
            test_loss += loss.item()

            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            test_pred_list.append(to_cpu_npy(pred))
            test_labels_list.append(to_cpu_npy(x_batch))

    test_scores = np.concatenate(test_pred_list)
    test_labels = np.concatenate(test_labels_list)

    return test_loss, test_scores, test_labels

    
max_epoch = 50
batch_size = 128 
time_point = 12.5
beta = 0.5 ### controls relative weight of reconstruction and KL loss, default = 0.5
cluster_weight_type = 'sigmoid' ### vanilla (no pi), sigmoid, or softmax
layer_size_determination = 'log' ### log, linear, or user, default = log
learning_rate = 1e-4

str_time_point = str(time_point).replace('.', '_')
base_path = os.getcwd()
# fname = os.path.join(base_path, 'cao2019', 'gene_exp_mat_time_' + \
#                      str_time_point + '_100k_sf_1e04_rc.mtx')
fname = "../Data/mouse_cell/w_preprocess/gene_cnt_mat_time_9_5.mtx"

    
    
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm_type = 'rc1e04' #log, rc1e04
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


sparse_mat = scipy.io.mmread(fname) #coo format
sparse_mat = coo_matrix.transpose(sparse_mat) #rows now correspond to samples
sparse_mat = coo_matrix.tocsr(sparse_mat) #easier to index
dense_mat = np.array(csr_matrix.todense(sparse_mat))

num_cells, num_genes = np.shape(sparse_mat)

train_size, valid_size = (int(0.7*num_cells), int(0.15*num_cells))
test_size = num_cells - train_size - valid_size

train_set, valid_set, test_set = \
    random_split(CustomDataset(dense_mat, dense_mat), \
    [train_size, valid_size, test_size])
            
train_set = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_set = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_set = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


if layer_size_determination == 'user':
    latent_size = 50
    layer_size_list = [num_genes, 1000, 500, latent_size]
    num_layers = int(len(layer_size_list) - 1)

elif layer_size_determination == 'linear':
    num_layers = 3
    latent_size = 50
    m = -(num_genes - latent_size)/num_layers
    b = num_genes
    
    layer_size_list = [num_genes]
    
    for i in np.arange(1, num_layers, 1):
        new_layer_size = int(m*i + b)
        layer_size_list.append(new_layer_size)  
    layer_size_list.append(latent_size)

elif layer_size_determination == 'log':
    latent_size = 32
    num_layers = 2
    
    a = num_genes
    b = -np.log2(num_genes/latent_size)/num_layers
    
    layer_size_list = [num_genes]
    
    for i in np.arange(1, num_layers, 1):
        new_layer_size = int(a*2**(b*i))
        # layer_size_list.append(new_layer_size)
        layer_size_list.append(new_layer_size)  
    layer_size_list.append(latent_size)

model = VAE(num_layers, layer_size_list, cluster_weight_type, beta).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

print("\n"+"Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

train_loss_list, valid_loss_list, = \
    train_model(model, optimizer, train_set, valid_set, max_epoch, device)

test_loss, test_scores, test_labels = eval_model(model, test_set, device)

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

test_pcc_str = str(np.round(test_pcc, 3))
test_scc_str = str(np.round(test_scc, 3))
cc_str = (f"Test PCC = {test_pcc_str}\n"
          f"Test SCC = {test_scc_str}")

scores_and_labels = np.concatenate((test_labels, test_scores), axis=0)
trans = umap.UMAP(random_state=0).fit(scores_and_labels)


epochs = np.arange(0, max_epoch, 1)


if cluster_weight_type == 'softmax':
    cluster_weight_type_str = 'Softmax Latent Weights'
elif cluster_weight_type == 'sigmoid':
    cluster_weight_type_str = 'Sigmoidal Latent Weights'
else:
    cluster_weight_type_str = 'Vanilla VAE Architecture'
    
mpl.rcdefaults()

fig = plt.figure(0, figsize = (10, 5))
plt.suptitle('Day ' + str(time_point) + ' with ' + str(cluster_weight_type_str), fontsize=16, y=0.95)
fig.subplots_adjust(top=0.8)
subtitle_fontsize = 14
axes_labels_fontsize = 12

ax1 = fig.add_subplot(121)
ax1.plot(epochs, train_loss_list, 'orange', label='Training')
ax1.plot(epochs, valid_loss_list, 'blue', label='Validation')
ax1.set_xlabel('Epochs', fontsize = axes_labels_fontsize)
ax1.set_ylabel('Loss', fontsize = axes_labels_fontsize)
ax1.set_title('Wasserstein Loss', fontsize = subtitle_fontsize)
ax1.legend()
ax1.text(0.1, 0.1, cc_str, horizontalalignment='left', verticalalignment='center', \
     transform=ax1.transAxes)


ax2 = fig.add_subplot(122)
num_cells_test = np.size(test_labels, 0)
ax2.scatter(trans.embedding_[:num_cells_test, 0], trans.embedding_[:num_cells_test, 1], color='blue', alpha=0.5, s=2, label='True')
ax2.scatter(trans.embedding_[num_cells_test:, 0], trans.embedding_[num_cells_test:, 1], color='orange', alpha=0.5, s=2, label='Simulated')
ax2.legend()
ax2.set_title('UMAP', fontsize = subtitle_fontsize)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('Dimension 1', fontsize = axes_labels_fontsize)
ax2.set_ylabel('Dimension 2', fontsize = axes_labels_fontsize)

plt.show()
mpl.rcdefaults()


elapsed = (time.time() - start_time)
elapsed_h = int(elapsed//3600)
elapsed_m = int((elapsed - elapsed_h*3600)//60)
elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
print('Elapsed time for training + UMAP: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))

