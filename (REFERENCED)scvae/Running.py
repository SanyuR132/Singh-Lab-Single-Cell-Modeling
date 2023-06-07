'''
Description:
    Train and evaluate the scVAE model.

Authot:
    Jiaqi Zhang
'''

from Models.scvae.defaults import defaults
from Models.scvae.cli import *
import json

from scipy.stats import pearsonr, spearmanr
import loompy
import numpy as np
import matplotlib.pyplot as plt
import umap





def loadArgs(args_filename):
    with open(args_filename) as file:
        args_dict = json.load(file)
    return args_dict


def train_model(args_dict):
    #TODO: maybe add other parameters
    model = train(**args_dict)
    return model


if __name__ == '__main__':
    args_dict = loadArgs("./splat_simulation_exp.json")
    model = train_model(args_dict)
    print()




# model = arguments.func(**vars(arguments))
# generated_data = model.sample(sample_size=1500)[0]
# generated_data_values = generated_data.values
#
# # # Save prediction
# # with loompy.connect("./gene_by_cell_testing_all.loom") as data_file:
# #     values = data_file[:, :].T
#
# # ------------------------------------------------------
# mean_gene_exp_pred = np.mean(generated_data_values, axis=0)
# mean_gene_exp_labels = np.mean(values, axis=0)
# test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
# test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
# print('Test PCC:', test_pcc)
# print('Test SCC:', test_scc)
#
# test_pcc_str = str(np.round(test_pcc, 3))
# test_scc_str = str(np.round(test_scc, 3))
# cc_str = (
#     f"Test PCC = {test_pcc_str}\n"
#     f"Test SCC = {test_scc_str}"
# )
#
# scores_and_labels = np.concatenate((values, generated_data_values), axis=0)
# # model = umap.UMAP().fit(values)
# model = umap.UMAP().fit(scores_and_labels)
# emedding = model.transform(scores_and_labels)
# plt.scatter(emedding[:1500, 0], emedding[:1500, 1], color='blue', alpha=0.5, s=5, label='True')
# plt.scatter(emedding[1500:, 0], emedding[1500:, 1], color='orange', alpha=0.5, s=5, label='Simulated')
# plt.legend()
# plt.title('UMAP', fontsize=15)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('Dimension 1', fontsize=15)
# plt.ylabel('Dimension 2', fontsize=15)
# plt.show()
