import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import mmread
from scipy.sparse import coo_matrix
from umap import UMAP
from sklearn.decomposition import PCA
import scanpy
import wot

# Load data
# day_num = [9, 10, 11, 12, 13]
day_num = [9, 10, 11]
data_list = []
day_list = []
for day in day_num:
    print("-" * 70)
    print("Read data at {} time point.".format(day))
    day_data = mmread("../../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_{}_5.mtx".format(day))
    day_data = np.asarray(day_data.todense()).T
    day_data = day_data[:(day-8)*100, :] # TODO: select 100 cells for efficiency
    data_list.append(day_data)
    day_list.extend([day for _ in range(day_data.shape[0])])
    print("Data shape : {}".format(day_data.shape))
all_data = np.concatenate(data_list, axis=0)
all_data = scanpy.AnnData(all_data)
all_data.obs["day"] = day_list
print("-" * 70)
print("All data shape : {}".format(all_data.shape))

# -----
print("=" * 70)
print("Construct wot model")
wot_model = wot.ot.OTModel(all_data, epsilon = 0.05, lambda1 = 0.1, lambda2 = 0.1, growth_iters=2)
print("Start fitting...")
tmap_pair = {}
tmap_pair[(9, 10)] = wot_model.compute_transport_map(9, 10)
tmap_pair[(10, 11)] = wot_model.compute_transport_map(10, 11)

# -----
# Construct transport map
print("=" * 70)
print("Construct transport map model")
meta_data = pd.DataFrame(data={"id":all_data.obs_names, "day":day_list})
tmap_model = wot.tmap.TransportMapModel(tmap_pair, meta_data)
couple_transport = tmap_model.get_coupling(9, 10)

df_data = all_data.to_df()
data_day9 = df_data.iloc[:100]
pred_day10 = data_day9.values.T @ couple_transport.X
print()

# -----
# Construct populations
population = tmap_model.population_from_ids([int(each) for each in list(all_data.obs_names)], at_time=10)
target_population = tmap_model.push_forward(*population, to_time=11)
descendants = tmap_model.descendants(*population)
print()
#
#
# -----
# Visualization
plt.scatter(tmap_pair[(9, 10)].obs['g1'],tmap_pair[(9, 10)].obs['g2'])
plt.xlabel("g1")
plt.ylabel("g2")
plt.title("Input vs Output Growth Rates")
plt.tight_layout()
plt.show()
#
# print()