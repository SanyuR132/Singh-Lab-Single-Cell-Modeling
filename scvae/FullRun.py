"""
Description:
    Model running for PBMC data.
 
Author:
    Jiaqi Zhang
"""

from Running import normalAugmentation
from Py_Utils import readH5ad, getTimeStr, addDefaultArg, getDefualtLayerSize, readMtx
import sys
import pickle
import argparse
import os
from scipy.sparse import csr_matrix
import numpy as np
import yaml
import json
from scipy.io import mmwrite
import datetime
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr, spearmanr
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
from datetime import datetime, date
import time


def plot_dist_shape_metrics(
    test_preds,
    test_labels,
    test_prev=np.array([]),
    metric_type_list=["KS"],
    stat_types_flag=1,
):
    if stat_types_flag == 1:
        stat_type_list = [
            "mean_gene",
            "var_gene",
            "mean_var_gene",
            "mean_cell",
            "var_cell",
            "mean_var_cell",
        ]
        stat_type_list_long = [
            "Gene Expression Mean",
            "Gene Expression Variance",
            "Gene Expression Mean/Variance Ratio",
            "Cell Expression Mean",
            "Cell Expression Variance",
            "Cell Expression Mean/Variance Ratio",
        ]
        (nrows, ncols) = (2, 3)

    if stat_types_flag == 2:
        stat_type_list = ["mean_gene", "var_gene", "mean_cell", "var_cell"]
        stat_type_list_long = [
            "Gene Expression Mean",
            "Gene Expression Variance",
            "Cell Expression Mean",
            "Cell Expression Variance",
        ]
        (nrows, ncols) = (2, 2)

    ks_dist_list = []
    ks_dist_list2 = []
    ks_dist_prev_list = []
    delta_auc_list = []
    delta_auc_prev_list = []
    pval_list = []
    pval_list2 = []

    for m in np.arange(len(metric_type_list)):
        metric_type = metric_type_list[m]

        fig, axes = plt.subplots(
            nrows, ncols, sharey=True, squeeze=False, figsize=(12, 8)
        )
        plt.rcParams.update({"figure.autolayout": True})
        # path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
        # prop = font_manager.FontProperties(fname=path)
        # mpl.rcParams['font.family'] = prop.get_name()
        # fontname = mpl.rcParams['font.family']

        fontsize_ax_labels = 16
        fontsize_ax_tick_labels = 14
        fontsize_legend = 12
        fontsize_title = 18
        fontsize_txt = 12
        alpha_true = 0.9
        alpha_gen = 0.65
        color_true = "dodgerblue"
        color_gen = "red"  #'firebrick'
        wspace = 1.0
        hspace = 1.0
        linewidth = 0.5

        if metric_type == "KS":
            counter = 0

            for r in np.arange(nrows):
                axis_along = r

                for c in np.arange(ncols):
                    ax = axes[r, c]
                    stat_type = stat_type_list[counter]
                    stat_type_str = stat_type_list_long[counter]

                    eps = 1e-6

                    if stat_type == "mean_gene":
                        stat_test_preds = np.mean(test_preds, axis=axis_along)
                        stat_test_labels = np.mean(test_labels, axis=axis_along)
                    elif stat_type == "var_gene":
                        stat_test_preds = np.var(test_preds, axis=axis_along)
                        stat_test_labels = np.var(test_labels, axis=axis_along)
                    elif stat_type == "mean_var_gene":
                        stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                            np.var(test_preds, axis=axis_along) + eps
                        )
                        stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                            np.var(test_labels, axis=axis_along) + eps
                        )
                    elif stat_type == "mean_cell":
                        stat_test_preds = np.mean(test_preds, axis=axis_along)
                        stat_test_labels = np.mean(test_labels, axis=axis_along)
                    elif stat_type == "var_cell":
                        stat_test_preds = np.var(test_preds, axis=axis_along)
                        stat_test_labels = np.var(test_labels, axis=axis_along)
                    elif stat_type == "mean_var_cell":
                        stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                            np.var(test_preds, axis=axis_along) + eps
                        )
                        stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                            np.var(test_labels, axis=axis_along) + eps
                        )

                    if test_prev.size != 0:
                        if stat_type == "mean_gene":
                            stat_test_prev = np.mean(test_prev, axis=axis_along)
                        elif stat_type == "var_gene":
                            stat_test_prev = np.var(test_prev, axis=axis_along)
                        elif stat_type == "mean_var_gene":
                            stat_test_prev = np.mean(test_prev, axis=axis_along) / (
                                np.var(test_prev, axis=axis_along) + eps
                            )
                        elif stat_type == "mean_cell":
                            stat_test_prev = np.mean(test_prev, axis=axis_along)
                        elif stat_type == "var_cell":
                            stat_test_prev = np.var(test_prev, axis=axis_along)
                        elif stat_type == "mean_var_cell":
                            stat_test_prev = np.mean(test_prev, axis=axis_along) / (
                                np.var(test_prev, axis=axis_along) + eps
                            )

                        stat_test_prev = stat_test_prev[~np.isnan(stat_test_prev)]
                        stat_test_prev = stat_test_prev[~np.isinf(stat_test_prev)]

                        ks_dist_prev, pval = ks_2samp(stat_test_prev, stat_test_labels)
                        ks_dist_prev_list.append(ks_dist_prev)

                    stat_test_preds = stat_test_preds[~np.isnan(stat_test_preds)]
                    stat_test_labels = stat_test_labels[~np.isnan(stat_test_labels)]
                    stat_test_preds = stat_test_preds[~np.isinf(stat_test_preds)]
                    stat_test_labels = stat_test_labels[~np.isinf(stat_test_labels)]

                    ks_dist, pval = ks_2samp(stat_test_preds, stat_test_labels)

                    num_bins = 50
                    max_bin_val = max(np.max(stat_test_preds), np.max(stat_test_labels))
                    # max_bin_val = max(np.percentile(stat_test_preds, 99), np.percentile(stat_test_labels, 99))
                    min_bin_val = 0
                    bins = np.linspace(min_bin_val, max_bin_val, num_bins)
                    half_bin_width = (bins[1] - bins[0]) / 2

                    pred_bars, pred_bin_edges = np.histogram(stat_test_preds, bins=bins)
                    label_bars, label_bin_edges = np.histogram(
                        stat_test_labels, bins=bins
                    )

                    ax.bar(
                        bins[:-1] + half_bin_width,
                        label_bars / np.size(stat_test_labels),
                        width=2 * half_bin_width,
                        color=color_true,
                        alpha=alpha_true,
                        label="True",
                        edgecolor="k",
                        linewidth=linewidth,
                    )
                    ax.bar(
                        bins[:-1] + half_bin_width,
                        pred_bars / np.size(stat_test_preds),
                        width=2 * half_bin_width,
                        color=color_gen,
                        alpha=alpha_gen,
                        label="Simulated",
                        edgecolor="k",
                        linewidth=linewidth,
                    )

                    if test_prev.size != 0:
                        prev_bars, prev_bin_edges = np.histogram(
                            stat_test_prev, bins=bins
                        )
                        ax.bar(
                            bins[:-1] + half_bin_width,
                            prev_bars / np.size(stat_test_prev),
                            width=2 * half_bin_width,
                            color="green",
                            alpha=0.25,
                            edgecolor="k",
                            label="Previous",
                        )

                    ax.text(
                        0.5,
                        0.5,
                        "2-sided KS = {0:.2f}".format(ks_dist),
                        transform=ax.transAxes,
                        fontsize=fontsize_txt,
                    )
                    ax.set_xlabel(stat_type_str, fontsize=fontsize_ax_labels)
                    ax.tick_params(
                        axis="x", labelsize=fontsize_ax_tick_labels
                    )  # , rotation=45)
                    ax.tick_params(axis="y", labelsize=fontsize_ax_tick_labels)
                    ax.legend(fontsize=fontsize_legend)
                    if c == 0:
                        ax.set_ylabel(
                            "Normalized Frequency", fontsize=fontsize_ax_labels
                        )

                    counter += 1

                    ks_dist_list.append(ks_dist)

                    pval_list.append(pval)

            plt.subplots_adjust(wspace=wspace, hspace=hspace)

        else:
            counter = 0

            for r in np.arange(nrows):
                axis_along = r

                for c in np.arange(ncols):
                    ax = axes[r, c]
                    stat_type = stat_type_list[counter]
                    stat_type_str = stat_type_list_long[counter]

                    eps = 1e-6

                    if stat_type == "mean_gene":
                        stat_test_preds = np.mean(test_preds, axis=axis_along)
                        stat_test_labels = np.mean(test_labels, axis=axis_along)
                    elif stat_type == "var_gene":
                        stat_test_preds = np.var(test_preds, axis=axis_along)
                        stat_test_labels = np.var(test_labels, axis=axis_along)
                    elif stat_type == "mean_var_gene":
                        stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                            np.var(test_preds, axis=axis_along) + eps
                        )
                        stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                            np.var(test_labels, axis=axis_along) + eps
                        )
                    elif stat_type == "mean_cell":
                        stat_test_preds = np.mean(test_preds, axis=axis_along)
                        stat_test_labels = np.mean(test_labels, axis=axis_along)
                    elif stat_type == "var_cell":
                        stat_test_preds = np.var(test_preds, axis=axis_along)
                        stat_test_labels = np.var(test_labels, axis=axis_along)
                    elif stat_type == "mean_var_cell":
                        stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                            np.var(test_preds, axis=axis_along) + eps
                        )
                        stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                            np.var(test_labels, axis=axis_along) + eps
                        )

                    if test_prev.size != 0:
                        if stat_type == "mean_gene":
                            stat_test_prev = np.mean(test_prev, axis=axis_along)
                        elif stat_type == "var_gene":
                            stat_test_prev = np.var(test_prev, axis=axis_along)
                        elif stat_type == "mean_var_gene":
                            stat_test_prev = np.mean(test_prev, axis=axis_along) / (
                                np.var(test_prev, axis=axis_along) + eps
                            )
                        elif stat_type == "mean_cell":
                            stat_test_prev = np.mean(test_prev, axis=axis_along)
                        elif stat_type == "var_cell":
                            stat_test_prev = np.var(test_prev, axis=axis_along)
                        elif stat_type == "mean_var_cell":
                            stat_test_prev = np.mean(test_prev, axis=axis_along) / (
                                np.var(test_prev, axis=axis_along) + eps
                            )

                        stat_test_prev = stat_test_prev[~np.isnan(stat_test_prev)]
                        stat_test_prev = stat_test_prev[~np.isinf(stat_test_prev)]

                    stat_test_preds = stat_test_preds[~np.isnan(stat_test_preds)]
                    stat_test_labels = stat_test_labels[~np.isnan(stat_test_labels)]
                    stat_test_preds = stat_test_preds[~np.isinf(stat_test_preds)]
                    stat_test_labels = stat_test_labels[~np.isinf(stat_test_labels)]

                    ks_dist, pval = ks_2samp(stat_test_preds, stat_test_labels)

                    ecdf_pred_func = ECDF(stat_test_preds)
                    ecdf_labels_func = ECDF(stat_test_labels)

                    # num_bins = int(np.sqrt(len(stat_test_labels)))
                    num_bins = 1000
                    max_bin_val = max(np.max(stat_test_preds), np.max(stat_test_labels))
                    min_bin_val = 0
                    bins = np.linspace(min_bin_val, max_bin_val, num_bins)
                    scaled_bins = bins / max_bin_val

                    ecdf_preds = ecdf_pred_func(bins)
                    ecdf_labels = ecdf_labels_func(bins)

                    delta_arr = np.array(
                        [
                            np.abs(ecdf_preds[i] - ecdf_labels[i])
                            for i in np.arange(num_bins)
                        ]
                    )
                    auc_delta = np.sum(delta_arr) / num_bins

                    ax.plot(
                        scaled_bins,
                        ecdf_labels,
                        color="dodgerblue",
                        linewidth=linewidth,
                        label="True",
                    )
                    ax.plot(
                        scaled_bins,
                        ecdf_preds,
                        color="red",
                        linewidth=linewidth,
                        label="Simulated",
                    )

                    # ax.fill_between(scaled_bins, 0, ecdf_preds, facecolor=color_gen, interpolate=True, alpha=alpha_gen, label='Generated')
                    # ax.fill_between(scaled_bins, 0, ecdf_labels, facecolor=color_true, interpolate=True, alpha=alpha_true, label='True')

                    ### ax.fill_between(bins, label_kde, pred_kde, where=pred_kde >= label_kde, facecolor='green', interpolate=True)
                    ### ax.fill_between(bins, pred_kde, label_kde, where=label_kde >= pred_kde, facecolor='red', interpolate=True)

                    if test_prev.size != 0:
                        ecdf_prev_func = ECDF(stat_test_prev)
                        ecdf_prev = ecdf_prev_func(bins)

                        delta_arr = np.array(
                            [
                                np.abs(ecdf_prev[i] - ecdf_labels[i])
                                for i in np.arange(num_bins)
                            ]
                        )
                        auc_delta_prev = np.sum(delta_arr) / num_bins

                        ax.plot(scaled_bins, ecdf_prev, color="green", label="Previous")
                        delta_auc_prev_list.append(auc_delta_prev)

                    ax.text(
                        0.7,
                        0.5,
                        "KS = {0:.3f}".format(ks_dist),
                        transform=ax.transAxes,
                        fontsize=fontsize_txt,
                    )
                    # ax.text(0.5, 0.5, "Delta = {0:.3f}".format(auc_delta), transform=ax.transAxes,
                    #         fontsize=fontsize_txt)
                    ax.set_xlabel(
                        "Scaled " + stat_type_str, fontsize=fontsize_ax_labels
                    )
                    ax.tick_params(
                        axis="x", labelsize=fontsize_ax_tick_labels
                    )  # , rotation=45)
                    ax.tick_params(axis="y", labelsize=fontsize_ax_tick_labels)
                    # ax.legend(fontsize = fontsize_legend)
                    if c == 0:
                        ax.set_ylabel("eCDF", fontsize=fontsize_ax_labels)

                    plt.subplots_adjust(wspace=wspace, hspace=hspace)

                    delta_auc_list.append(auc_delta)
                    ks_dist_list.append(ks_dist)
                    pval_list.append(pval)

                    counter += 1

    # if test_prev.size == 0:
    #     df_stats = pd.DataFrame(np.array([np.round(ks_dist_list, 3), np.round(delta_auc_list, 3)]).T, \
    #         columns = ['KS', 'Scaled Area Difference'], index=stat_type_list)
    # else:
    #     df_stats = pd.DataFrame(np.array([np.round(ks_dist_list, 3), np.round(ks_dist_prev_list, 3), \
    #         np.round(delta_auc_list, 3), np.round(delta_auc_prev_list, 3)]).T, \
    #         columns = ['KS-pred', 'Delta AUC-pred', 'KS-prev', 'Delta AUC-prev'], index=stat_type_list)

    # print(len(ks_dist_list2))
    # print(len(pval_list2))

    if test_prev.size == 0:
        df_stats = pd.DataFrame(
            np.array([np.round(ks_dist_list, 3), np.round(pval_list, 3)]).T,
            columns=["KS", "p"],
            index=stat_type_list,
        )
    else:
        df_stats = pd.DataFrame(
            np.array([np.round(ks_dist_list, 3), np.round(ks_dist_prev_list, 3)]).T,
            columns=["KS-pred", "KS-prev"],
            index=stat_type_list,
        )

    print("KS performance (sum):", np.round(sum(ks_dist_list), 3), "\n")
    print(df_stats)

    return df_stats, sum(ks_dist_list), pval_list, fig


def plot_cc_metrics(test_preds, test_labels, test_prev=np.array([])):
    mean_gene_exp_pred = np.mean(test_preds, axis=0)
    mean_gene_exp_labels = np.mean(test_labels, axis=0)
    test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
    test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)

    stats_dict = {"test_pcc": test_pcc, "test_scc": test_scc}
    if np.size(test_prev) == 0:
        fig, ax = plt.subplots(1, 1, sharey=True)
        plt.rcParams.update({"figure.autolayout": True})
        # path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
        # prop = font_manager.FontProperties(fname=path)
        # mpl.rcParams['font.family'] = prop.get_name()
        # fontname = mpl.rcParams['font.family']

        fontsize_ax_labels = 14
        fontsize_xtick_labels = 14
        fontsize_ytick_labels = 14
        fontsize_legend = 10
        fontsize_title = 16
        fontsize_text = 10

        ax.scatter(mean_gene_exp_labels, mean_gene_exp_pred)
        ax.text(
            0.65,
            0.25,
            "PCC = {0:.3f}\nSCC = {1:.3f}".format(test_pcc, test_scc),
            transform=ax.transAxes,
            fontsize=fontsize_text,
        )

        print("Test PCC:", test_pcc)
        print("Test SCC:", test_scc)

    elif np.size(test_prev) != 0:
        mean_gene_exp_baseline = np.mean(test_prev, axis=0)
        baseline_pcc, _ = pearsonr(mean_gene_exp_baseline, mean_gene_exp_labels)
        baseline_scc, _ = spearmanr(mean_gene_exp_baseline, mean_gene_exp_labels)

        fig, axes = plt.subplots(1, 2, sharey=True)
        plt.rcParams.update({"figure.autolayout": True})
        # path = 'C:\\Windows\\Fonts\\LiberationSerif-Italic.ttf'
        # prop = font_manager.FontProperties(fname=path)
        # mpl.rcParams['font.family'] = prop.get_name()
        # fontname = mpl.rcParams['font.family']

        fontsize_ax_labels = 14
        fontsize_xtick_labels = 14
        fontsize_ytick_labels = 14
        fontsize_legend = 10
        fontsize_title = 16
        fontsize_text = 10

        ax = axes[0]
        ax.scatter(mean_gene_exp_labels, mean_gene_exp_pred, color="orange")
        ax.text(
            0.65,
            0.25,
            "PCC = {0:.3f}\nSCC = {1:.3f}".format(test_pcc, test_scc),
            transform=ax.transAxes,
            fontsize=fontsize_text,
        )

        print("Test PCC:", test_pcc)
        print("Test SCC:", test_scc)

        ax = axes[1]
        ax.scatter(mean_gene_exp_labels, mean_gene_exp_baseline, color="green")
        ax.text(
            0.65,
            0.25,
            "PCC = {0:.3f}\nSCC = {1:.3f}".format(baseline_pcc, baseline_scc),
            transform=ax.transAxes,
            fontsize=fontsize_text,
        )

        print("Test PCC:", baseline_pcc)
        print("Test SCC:", baseline_scc)

        stats_dict["baseline_pcc"] = baseline_pcc
        stats_dict["baseline_scc"] = baseline_scc

    # CHANGE: return plot

    return pd.DataFrame(stats_dict, index=list(range(len(stats_dict)))), fig


# ------------------------------------------------------------------------------------
# Augmentation for different sampling ratios

# def clusterAugmentExp():
#     # parameters configuration
#     num_genes = 17789
#     latent_size = 32
#     num_layers = 2
#     layer_size_list = getDefualtLayerSize(num_genes, num_layers, latent_size)
#     date_str = getTimeStr()
#     config = {
#         "max_epoch": 50,
#         # "max_epoch" : 2,
#         "batch_size": 128,
#         "beta": 0.5,  # controls relative weight of reconstruction and KL loss, default = 0.5
#         "cluster_weight_type": 'sigmoid',  # vanilla (no pi), sigmoid, or softmax
#         "layer_size_list": layer_size_list,
#         "num_layers": num_layers,
#         # "learning_rate": 1e-4,
#         "learning_rate": 1e-4,
#     }
#     config = addDefaultArg(config)
#     # ------------------------------------------------
#     cluster_ls = [3]
#     cluster_size = {1:14123, 2:4716, 3:4574}
#     for c in cluster_ls:
#         print("=" * 70)
#         print("Cluster {}".format(c))
#         config["data"] = readH5ad("../../Data/PBMC/clusters/cluster{}_data-{}.h5ad".format(c, cluster_size[c]))
#         for s in [0.5, 0.25, 0.1, 0.05, 0.03, 0.01]: # different sampling ratio of training data
#             print("*" * 70)
#             print("Train data size : {}".format(s))
#             config["train_size"] = s
#             for t in range(5): # 5 trials
#                 print("#" * 70)
#                 print("TRIAL {}".format(t))
#                 config["model_save_path"] = \
#                     "../../Prediction/StructureVAE/PBMC_cluster{}-augmented-trial{}-VAE_model-{}.pt".format(
#                         c, t, s)
#                 config["prediction_save_path"] = \
#                     "../../Prediction/StructureVAE/PBMC_cluster{}-augmented-trial{}-VAE_estimation-{}.npy".format(
#                         c, t, s)
#                 clusterAugmentation(config)

# ------------------------------------------------------------------------------------
# Train, validate, test


# def normalAugmentExp(hyperparameters_file):
#     # parameters configuration
#     config = {
#         "config_file": "./DrosophilaData_exp.json",
#         "need_save": True,
#         # ------------------------------------------
#         # "prediction_save_path": "../../Prediction/scVAE/mouse_cell-scVAE_estimation.npz",
#         "test_preds_filename": "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/baseline_results/scVAE/drosophila/drosophila-scVAE_test_preds.mtx",
#     }

#     hyperparameter_dict = None
#     if hyperparameters_file:
#         with open(hyperparameters_file, "wb") as f:
#             hyperparameter_dict = pickle.load(f)

#     normalAugmentation(config, hyperparameter_dict)


parser = argparse.ArgumentParser()
parser.add_argument("-lhsdt", type=str)
parser.add_argument("-ttp", type=int)
parser.add_argument("-stu", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--hyperparameters_dir", type=str)
args = parser.parse_args()

if args.stu == "drosophila":
    config_file = "./DrosophilaData_exp.json"
    data_path = os.path.join(args.data_dir, "drosophila", "upto_tp" + str(args.ttp))

if args.stu == "WOT":
    config_file = "./WOTData_exp.json"
    # Change data path based on system
    data_path = os.path.join(args.data_dir, "WOT", "upto_tp" + str(args.ttp))


elif args.stu == "mouse_cortex1_chromium":
    config_file = "./MouseCortex1Chromium_exp.json"
    data_path = os.path.join(args.data_dir, "mouse_cortex1_chromium", "upto_tp0")


elif args.stu == "mouse_cortex2_chromium":
    config_file = "./MouseCortex2Chromium_exp.json"
    data_path = data_path = os.path.join(
        args.data_dir, "mouse_cortex2_chromium", "upto_tp0"
    )


elif args.stu == "mouse_cortex1_smart_seq":
    config_file = "./MouseCortex1Chromium_exp.json"
    data_path = data_path = os.path.join(
        args.data_dir, "mouse_cortex1_smart_seq", "upto_tp0"
    )


elif args.stu == "mouse_cortex2_smart_seq":
    config_file = "./MouseCortex1Chromium_exp.json"
    data_path = data_path = os.path.join(
        args.data_dir, "mouse_cortex2_smart_seq", "upto_tp0"
    )

elif args.stu == "PBMC":
    config_file = "./MouseCortex2SmartSeq_exp.json"
    data_path = (args.data_dir, "PBMC", "upto_tp0")

train_file_path = os.path.join(data_path, "train_data.mtx")
test_file_path = os.path.join(data_path, "test_data.mtx")

test_labels = csr_matrix(readMtx(test_file_path)).T
test_labels_npy = np.array(test_labels.todense())
train_data = csr_matrix(readMtx(train_file_path)).T

with open(config_file) as f:
    args_dict = json.load(f)

with open(os.path.join(data_path, "cell_clusters_dict.yml"), "r") as f:
    cell_clusters_dict = yaml.load(f, Loader=yaml.Loader)

print("test data shape:", test_labels.shape)
print("train data shape:", train_data.shape)

today = date.today()
mdy = today.strftime("%Y-%m-%d")
clock = datetime.now()
hms = clock.strftime("%H-%M-%S")
hm = clock.strftime("%Hh-%Mm")
hm_colon = clock.strftime("%H:%M")
date_and_time = mdy + "-at-" + hms
start_time = time.time()

config = {
    "need_save": True,
    # ------------------------------------------
    # "prediction_save_path": "../../Prediction/scVAE/mouse_cell-scVAE_estimation.npz",
    # "test_preds_filename": args.save_dir,
}

if args.lhsdt:
    hyperparameters_file_path = os.path.join(
        args.hyperparameters_dir,
        "scVAE",
        args.stu,
        args.lhsdt + "_ttp_" + str(args.ttp),
        "hyperparameters.pt",
    )

    with open(hyperparameters_file_path, "rb") as f:
        hyperparameter_dict = pickle.load(f)

    args_dict.update(hyperparameter_dict)


test_preds = normalAugmentation(
    args_dict, train_file_path, test_file_path, cell_clusters_dict
)

print("test labels shape:", test_labels_npy.shape)
print("test preds shape:", test_preds.shape)

save_dir = os.path.join(args.save_dir, date_and_time + "_ttp_" + str(args.ttp))

if args.lhsdt:
    save_dir += "_tuned"

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

print("save dir:", save_dir)
print("dir exists?", os.path.isdir(save_dir))

df_stats, _, _, ks_plot = plot_dist_shape_metrics(test_preds, test_labels_npy)
df_stats.to_csv(os.path.join(save_dir, "ks_stats.csv"))
ks_plot.savefig(os.path.join(save_dir, "ks_plots.png"))

cc_stats, cc_plot = plot_cc_metrics(test_preds, test_labels_npy)
cc_stats.to_csv(os.path.join(save_dir, "cc_stats.csv"))
cc_plot.savefig(os.path.join(save_dir, "cc_plots.png"))

## no longer saving predictions to reduce disk usage
# mmwrite(os.path.join(save_dir, "test_preds.mtx"), test_preds)

with open(os.path.join(save_dir, "info.txt"), "w") as f:
    f.write("Model reference date and time: " + date_and_time + "\n")
    f.write("\n")
    f.write("Truncated time point (included in training): " + str(args.ttp) + "\n")
    f.write(f"Tuned?: {bool(args.lhsdt)}" + "\n")
    if args.lhsdt:
        f.write("hyperparameters reference date and time: " + str(args.lhsdt) + "\n")
        f.write("hyperparameters (default if not specified):" + "\n")
        for param, val in hyperparameter_dict.items():
            f.write(f"{param}: {val}")

print("finished saving")

