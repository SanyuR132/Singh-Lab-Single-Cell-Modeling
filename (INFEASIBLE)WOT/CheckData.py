import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import mmread
from scipy.sparse import coo_matrix
from umap import UMAP
from sklearn.decomposition import PCA
import wot

# Load data
day_num = [9, 10, 11, 12, 13]
# day_num = [9, 10]
data_list = []
day_list = []
for day in day_num:
    print("-" * 70)
    print("Read data at {} time point.".format(day))
    day_data = mmread("../../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_{}_5.mtx".format(day))
    day_data = np.asarray(day_data.todense()).T
    data_list.append(day_data)
    day_list.extend([day for _ in range(day_data.shape[0])])
    print("Data shape : {}".format(day_data.shape))
all_data = np.concatenate(data_list, axis=0)
print("-" * 70)
print("All data shape : {}".format(all_data.shape))

# Dimensionality reduction
print("=" * 70)
print("Start umap...")
embedded_data = UMAP(n_components=2).fit_transform(all_data)
print("Embedded data shape : {}".format(embedded_data.shape))

# Visualization
figure = plt.figure()
plt.axis('off')
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=day_list,
               s=8, marker=',', edgecolors='none', alpha=0.8)
cb = plt.colorbar()
cb.ax.set_title('Day')
plt.tight_layout()
plt.savefig("mouse_cell-data_vis.pdf")
plt.show()
plt.close()