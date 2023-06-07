import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import mmread
import umap
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import pearsonr



# -----
# true_train_data = np.concatenate([
#     mmread("../../Data/mouse_cell/w_preprocess/gene_cnt_mat_time_9_5.mtx").todense().T,
#     mmread("../../Data/mouse_cell/w_preprocess/gene_cnt_mat_time_10_5.mtx").todense().T
# ], axis=0)
true_train_data = np.concatenate([
    mmread("../../Data/WOT/pt_data/day2_data.mtx").todense(),
    mmread("../../Data/WOT/pt_data/day2.5_data.mtx").todense(),
    mmread("../../Data/WOT/pt_data/day3_data.mtx").todense(),
    mmread("../../Data/WOT/pt_data/day3.5_data.mtx").todense(),
    # mmread("../../Data/WOT/pt_data/day9_data.mtx").todense()
], axis=0)
print("True train data shape : {}".format(true_train_data.shape))
scaler = sklearn.preprocessing.StandardScaler()
standard_true_train_data = scaler.fit_transform(true_train_data)
pca = PCA(n_components=50)
embed_true_train_data = pca.fit_transform(standard_true_train_data)
print("Complete fitting on train data")
# ----
# label_data = mmread("../../Data/mouse_cell/w_preprocess/gene_cnt_mat_time_11_5.mtx").todense().T
label_data = mmread("../../Data/WOT/pt_data/day4_data.mtx").todense()
print("Label data shape : {}".format(label_data.shape))
#
#
#
# vis_data = PCA(n_components=2).fit_transform(np.concatenate([true_train_data, label_data]))
# # vis_data = umap.UMAP().fit_transform(np.concatenate([label_data, simulated_data_second]))
# print("Visualization data shape : {}".format(vis_data.shape))
# vis_train_data = vis_data[:true_train_data.shape[0]]
# vis_label_data = vis_data[true_train_data.shape[0]:]
# plt.scatter(vis_train_data[:, 0], vis_train_data[:, 1], s=10, c="g", alpha=0.3, label="day 8 ~ day 9")
# plt.scatter(vis_label_data[:, 0], vis_label_data[:, 1], s=10, c="r", alpha=0.3, label="day 9.5")
# plt.legend(frameon=True, fontsize=13)
# plt.show()


# ----
#TODO: notice this prediction file
# with open("../../Prediction/PRESCIENT/mouse_cell/seed_2_train.sims_1_num.cells_10000_num.steps_1_subsets_None_None_simulation.pkl", "rb") as file:
with open("../../Prediction/PRESCIENT/seed_2_train.sims_1_num.cells_8000_num.steps_1_subsets_None_None_simulation.pkl", "rb") as file:
    simulated_data = pkl.load(file)[0]
simulated_data_first = scaler.inverse_transform(pca.inverse_transform(simulated_data[0]))
simulated_data_second = scaler.inverse_transform(pca.inverse_transform(simulated_data[0]))
print("Simulated data shape : {}".format(simulated_data_second.shape))

# ----------------------------------------------------------------
# cell_sparsity = [1 - len(np.where(label_data[i] != 0)[0]) / label_data.shape[1] for i in range(label_data.shape[0])]
# gene_sparsity = [1 - len(np.where(label_data[:,i] != 0)[0]) / label_data.shape[0] for i in range(label_data.shape[1])]
# plt.hist(np.asarray(cell_sparsity).squeeze(), color="r", alpha=0.5, label="Cell Sparsity", bins=20)
# plt.hist(np.asarray(gene_sparsity).squeeze(), color="b", alpha=0.5, label="Gene Sparsity", bins=20)
# plt.legend()
# plt.show()

true_cell_avg = np.mean(label_data, axis=1)[:8000]
true_cell_std = np.std(label_data, axis=1)[:8000]
pred_cell_avg = np.mean(simulated_data_second, axis=1)#[:3849]
pred_cell_std = np.std(simulated_data_second, axis=1)#[:3849]

true_gene_avg = np.mean(label_data, axis=0)
true_gene_std = np.std(label_data, axis=0)
pred_gene_avg = np.mean(simulated_data_second, axis=0)
pred_gene_std = np.std(simulated_data_second, axis=0)

# tmp = np.asarray(true_gene_avg).squeeze()
# large_idx = np.where(tmp > 0.8)
# print()

plt.subplot(2, 2, 1)
plt.title("Cell Avg. Expression [PCC = {:.2f}]".format(pearsonr(np.asarray(true_cell_avg).squeeze(), np.asarray(pred_cell_avg).squeeze())[0]))
plt.hist(np.asarray(true_cell_avg).squeeze(), color="r", alpha=0.5, label="True Data")
plt.hist(np.asarray(pred_cell_avg).squeeze(), color="b", alpha=0.5, label="Model Prediction")
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Cell Std. Expression [PCC = {:.2f}]".format(pearsonr(np.asarray(true_cell_std).squeeze(), np.asarray(pred_cell_std).squeeze())[0]))
plt.hist(np.asarray(true_cell_std).squeeze(), color="r", alpha=0.5, label="True Data")
plt.hist(np.asarray(pred_cell_std).squeeze(), color="b", alpha=0.5, label="Model Prediction")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Gene Avg. Expression [PCC = {:.2f}]".format(pearsonr(np.asarray(true_gene_avg).squeeze(), np.asarray(pred_gene_avg).squeeze())[0]))
plt.hist(np.asarray(true_gene_avg).squeeze(), color="r", alpha=0.5, label="True Data")
plt.hist(np.asarray(pred_gene_avg).squeeze(), color="b", alpha=0.5, label="Model Prediction")
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Gene Std. Expression [PCC = {:.2f}]".format(pearsonr(np.asarray(true_gene_std).squeeze(), np.asarray(pred_gene_std).squeeze())[0]))
plt.hist(np.asarray(true_gene_std).squeeze(), color="r", alpha=0.5, label="True Data")
plt.hist(np.asarray(pred_gene_std).squeeze(), color="b", alpha=0.5, label="Model Prediction")
plt.legend()

plt.tight_layout()
plt.show()
plt.close()


# -----
# vis_data = umap.UMAP().fit_transform(np.concatenate([true_train_data, label_data, simulated_data_second]))
vis_data = PCA(n_components=2).fit_transform(np.concatenate([true_train_data, label_data, simulated_data_second]))
# vis_data = umap.UMAP().fit_transform(np.concatenate([label_data, simulated_data_second]))
print("Visualization data shape : {}".format(vis_data.shape))
vis_train_data = vis_data[:true_train_data.shape[0]]
vis_label_data = vis_data[true_train_data.shape[0]:true_train_data.shape[0]+label_data.shape[0]]
vis_pred_data = vis_data[true_train_data.shape[0]+label_data.shape[0]:]

# vis_label_data = vis_data[:label_data.shape[0]]
# vis_pred_data = vis_data[label_data.shape[0]:]
#-----
# plt.scatter(vis_train_data[:, 0], vis_train_data[:, 1], s=10, c="g", alpha=0.3, label="day 8 ~ day 9")
# plt.scatter(vis_label_data[:, 0], vis_label_data[:, 1], s=10, c="r", alpha=0.3, label="true day 9.5")
# plt.scatter(vis_pred_data[:, 0], vis_pred_data[:, 1], s=10, c="b", alpha=0.3, label="simulated day 9.5")

plt.scatter(vis_train_data[:, 0], vis_train_data[:, 1], s=10, c="g", alpha=0.3, label="train data")
plt.scatter(vis_label_data[:, 0], vis_label_data[:, 1], s=10, c="r", alpha=0.3, label="true label")
plt.scatter(vis_pred_data[:, 0], vis_pred_data[:, 1], s=10, c="b", alpha=0.3, label="simulated data")

plt.legend(frameon=True, fontsize=13)
plt.show()