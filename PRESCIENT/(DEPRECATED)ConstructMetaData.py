import pandas as pd
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix
import scanpy

def mouseData():
    day_data_mat = []
    cell_id_list = []
    cell_day_list = []
    cell_type = []
    for idx, day in enumerate(["9", "10"]):
        filename = "../../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_{}_5.mtx".format(day)
        tmp_mat = mmread(filename)
        tmp_mat = tmp_mat.todense().T
        num_cells, num_genes = tmp_mat.shape
        tmp_cell_id = ["day{}_cell{}".format(day, i) for i in range(num_cells)]
        cell_id_list.extend(tmp_cell_id)
        cell_day_list.extend([int(idx) for i in range(num_cells)])
        cell_type.extend(["cell" for i in range(num_cells)])
        # -----
        tmp_mat_dict = {"id": tmp_cell_id}
        for i in range(num_genes):
            tmp_mat_dict["gene_{}".format(i)] = np.asarray(tmp_mat[:, i]).squeeze()
        tmp_mat = pd.DataFrame(data=tmp_mat_dict)
        day_data_mat.append(tmp_mat)
    # -----
    all_data = pd.concat(day_data_mat, axis=0)
    meta_data = pd.DataFrame(data={"id": cell_id_list, "timepoint": cell_day_list, "cell_type": cell_type})
    print("All data shape : {}".format(all_data.shape))
    print("Meta data shape : {}".format(meta_data.shape))
    # -----
    birth_gene_signature = pd.DataFrame(data={
        "id": np.arange(num_genes),
        "gene_symbol": ["gene_{}".format(i) for i in range(num_genes)]
    })
    death_gene_signature = birth_gene_signature.copy()
    # -----
    all_data = all_data.set_index("id")
    all_data.to_csv("test_mouse_all_data.csv")
    meta_data = meta_data.set_index("id")
    meta_data.to_csv("test_mouse_meta_data.csv")
    birth_gene_signature.to_csv("test_mouse_birth_gene_signature.csv")
    death_gene_signature.to_csv("test_mouse_death_gene_signature.csv")
    print("Finish saving data.")


def WOTData():
    full_data = scanpy.read_h5ad("../../Data/WOT/pt_data/ExprMatrix.h5ad")
    print("All data shape (cell, gene) : ", full_data.shape)
    res = scanpy.pp.highly_variable_genes(full_data, n_top_genes=2000, inplace=False)
    highly_gene_idx = np.where(res.highly_variable == True)[0]
    full_data = full_data[:, highly_gene_idx]
    print("Filtered data shape (cell, gene) : ", full_data.shape)
    # -----
    cell_days = pd.read_csv("../../Data/WOT/pt_data/cell_days.txt", index_col='id', sep='\t')
    unique_days = np.sort(cell_days.day.unique())
    print("The num of days = {}".format(len(unique_days)))
    day_data_mat = []
    cell_id_list = []
    cell_day_list = []
    cell_type = []
    for idx, day in enumerate([2, 2.5, 3, 3.5]):
        tmp_mat = full_data[cell_days[cell_days.day == day].index].to_df().values
        print("Data shape at day {} : {}".format(day, tmp_mat.shape))
        mmwrite("../../Data/WOT/pt_data/day{}_data.mtx".format(day), coo_matrix(tmp_mat))
        # -----
        num_cells, num_genes = tmp_mat.shape
        tmp_cell_id = ["day{}_cell{}".format(day, i) for i in range(num_cells)]
        cell_id_list.extend(tmp_cell_id)
        cell_day_list.extend([int(idx) for i in range(num_cells)])
        cell_type.extend(["cell" for i in range(num_cells)])
        # -----
        tmp_mat_dict = {"id": tmp_cell_id}
        for i in range(num_genes):
            tmp_mat_dict["gene_{}".format(i)] = np.asarray(tmp_mat[:, i]).squeeze()
        tmp_mat = pd.DataFrame(data=tmp_mat_dict)
        day_data_mat.append(tmp_mat)
    # -----
    all_data = pd.concat(day_data_mat, axis=0)
    meta_data = pd.DataFrame(data={"id": cell_id_list, "timepoint": cell_day_list, "cell_type": cell_type})
    print("All data shape : {}".format(all_data.shape))
    print("Meta data shape : {}".format(meta_data.shape))
    # -----
    birth_gene_signature = pd.DataFrame(data={
        "id": np.arange(num_genes),
        "gene_symbol": ["gene_{}".format(i) for i in range(num_genes)]
    })
    death_gene_signature = birth_gene_signature.copy()
    # -----
    tmp_mat = full_data[cell_days[cell_days.day == 4].index].to_df().values
    print("Label data shape : {}".format(tmp_mat.shape))
    mmwrite("../../Data/WOT/pt_data/day{}_data.mtx".format(4), coo_matrix(tmp_mat))
    # -----
    all_data = all_data.set_index("id")
    all_data.to_csv("../../Data/WOT/pt_data/WOT_all_data.csv")
    meta_data = meta_data.set_index("id")
    meta_data.to_csv("../../Data/WOT/pt_data/WOT_meta_data.csv")
    birth_gene_signature.to_csv("../../Data/WOT/pt_data/WOT_birth_gene_signature.csv")
    death_gene_signature.to_csv("../../Data/WOT/pt_data/WOT_death_gene_signature.csv")
    print("Finish saving data.")


if __name__ == '__main__':
    # mouseData()
    WOTData()