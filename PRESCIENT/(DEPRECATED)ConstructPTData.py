import pandas as pd
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix
import scanpy
import argparse
from prescient.commands.process_data import main as construct_pt_data


def create_parser():
    parser = argparse.ArgumentParser()
    # file I/0
    parser.add_argument('-d', '--data_path', type=str, default= "../../Data/WOT/pt_data/WOT_all_data.csv", required=False,
    help="Path to dataframe of expression values.")
    parser.add_argument('-o', '--out_dir', type=str, default= "../../Data/WOT/pt_data/", required=False,
    help="Path to output directory to store final PRESCIENT data file.")
    parser.add_argument('-m', '--meta_path', type=str, default= "../../Data/WOT/pt_data/WOT_meta_data.csv", required=False,
    help="Path to metadata containing timepoint and celltype annotation data.")

    # column names
    parser.add_argument('--tp_col', type=str, default="timepoint", required=False,
    help="Column name of timepoint feature in metadate provided as string.")
    parser.add_argument('--celltype_col', type=str, default="cell_type", required=False,
    help="Column name of celltype feature in metadata provided as string.")

    # dimensionality reduction growth_parameters
    parser.add_argument('--num_pcs', type=int, default=50, required=False,
    help="Define number of PCs to compute for input to training.")
    parser.add_argument('--num_neighbors_umap', type=int, default=10, required=False,
    help="Define number of neighbors for UMAP trasformation (UMAP used only for visualization.)")

    # proliferation scores
    parser.add_argument('--growth_path', type=str, default= "../../Data/WOT/pt_data/WOT_growth_data.pt",
    help="Path to torch pt file containg pre-computed growth weights. See vignette notebooks for generating growth rate vector.")
    return parser


def WOTData():
    data_args = create_parser().parse_args()
    construct_pt_data(data_args)


if __name__ == '__main__':
    WOTData()