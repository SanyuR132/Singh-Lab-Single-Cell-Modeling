import prescient.utils
import prescient
import pandas as pd
import numpy as np
import sklearn
import umap
import matplotlib.pyplot as plt



print("=" * 70)
print("Expression data :")
expr = pd.read_csv("../../Data/WOT/pt_data/WOT_all_data.csv", index_col=0)
print(expr.head())
print("=" * 70)
print("Meta data :")
metadata = pd.read_csv("../../Data/WOT/pt_data/WOT_meta_data.csv", index_col = 0)
print(metadata.head())
# -----
print("=" * 70)
scaler = sklearn.preprocessing.StandardScaler()
xs = pd.DataFrame(scaler.fit_transform(expr), index = expr.index, columns = expr.columns)
pca = sklearn.decomposition.PCA(n_components = 30)
xp_ = pca.fit_transform(xs)
print("Finished PCA")
# -----
print("=" * 70)
g, g_l=prescient.utils.get_growth_weights(xs, xp_, metadata, tp_col="cell_type", genes=list(expr.columns),
                   birth_gst="../../Data/Veres2019/hs_birth_msigdb_kegg.csv",
                   death_gst="../../Data/Veres2019/hs_death_msigdb_kegg.csv",
                   outfile="../../Data/WOT/pt_data/WOT_growth_data.pt"
                  )
print("Finished computing growth rate.")
# # -----
# um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = 30)
# xu = um.fit_transform(xp_)
# fig, ax = plt.subplots(figsize = (6,6))
# c = np.exp(g)
# ci = np.argsort(c)
# sax = ax.scatter(-xu[ci,0], xu[ci,1], s = 1, c = c[ci])
# plt.colorbar(sax, shrink = 0.9)
# plt.show()