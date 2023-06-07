# Structure VAE Model Settings

Experiment settings of structure VAE model on different datasets. We use the following parameter settings for all the dataset as follows.

|      Parameter      |      Value      |                                              Description                                              |
|:-------------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|
|      max_epoch      |        50       |                                       The number of VAE epochs.                                       |
|      batch_size     |       128       |                                      Batch size for VAE training.                                     |
|         beta        |       0.5       |                              Trade-off between reconstruction and KL oss.                             |
| cluster_weight_type |    "sigmoid"    |                                   Type of weights for cell clusters.                                  |
|    learning_rate    |       1e-4      |                                           VAE learning rate.                                          |
|      num_layers     |        2        |                                       The number of VAE layers.                                       |
|     latent_size     |        32       |                                       Size of latent variables.                                       |
|   layer_size_list   | \ | Size of each layer. This is automatically determined by input    data size and latent variable size.  |

-----------------------------------

## Splat Simulation Data

Training and simulation on the splat simulation data are in [SplatDataExp.py](SplatDataExp.py). For this dataset, the
`layer_size_list = [2000, 252, 32]`.

### Overall Augmentation

First, we split all the data into training:validation:testing=70%:15%:15% groups, train on only training data and do 
augmentation.   

### Cell Cluster Augmentation

For each cell cluster, randomly sampling 50%, 25%, 10%, 5%, 3%, and 1% of cells for model training and let the trained 
model augment this cluster of data. For the splat simulation data, we have three cell clusters with 3008, 5008, and 1984
cells respectively.  

-----------------------------------

## Mouse Cell Data

### Overall Augmentation

### Cell Cluster Augmentation

-----------------------------------

## PBMC Data

### Overall Augmentation

### Cell Cluster Augmentation