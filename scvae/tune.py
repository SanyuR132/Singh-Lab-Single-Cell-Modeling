"""
Description:
    Automatically tune model parameters.

Author:
    Jiaqi Zhang
"""
from Running import modelTrainForCV as scVAEEstimator
import argparse
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_contour
from Py_Utils import (
    readMtx,
    getTimeStr,
    addDefaultArg,
    getDefualtLayerSize,
    prepareDataset,
    readH5ad,
)
import optuna
from scipy.stats import ks_2samp, pearsonr
from scipy.sparse import csr_matrix
import numpy as np
import scanpy
import joblib
import anndata as ad
import yaml
import pickle

import time
import sys
import os
from datetime import datetime, date
import shutil

# ----------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "-stu", default="WOT", type=str, help="Which dataset to use",
)
parser.add_argument("-ttp", type=int)
parser.add_argument(
    "-s", default="KS", type=str,
)
parser.add_argument(
    "-n", "--num_trials", default=200, type=int, help="The number of trials."
)
parser.add_argument(
    "-tuti", "--tuning_time", type=int, help="Number of minutes to tune for"
)

parser.add_argument("--tune_all", action="store_true", default=False)
parser.add_argument("--tune_model_type", action="store_true", default=False)
parser.add_argument(
    "--tune_reconstruction_distribution", action="store_true", default=False
)
parser.add_argument(
    "--tune_prior_probabilities_method", action="store_true", default=False
)
parser.add_argument("--tune_latent_size", action="store_true", default=False)
parser.add_argument("--tune_hidden_sizes", action="store_true", default=False)
parser.add_argument(
    "--tune_number_of_warm_up_epochs", action="store_true", default=False
)
parser.add_argument("--tune_kl_weight", action="store_true", default=False)
parser.add_argument("--latent_size_upper_bound", type=int, default=64)

parser.add_argument("--save_dir", type=str)
parser.add_argument("--data_dir", type=str)

args = parser.parse_args()


def calc_ks(test_preds, test_labels):
    eps = 1e-6
    stats_list = [
        "mean_gene",
        "var_gene",
        "mean_var_gene",
        "mean_cell",
        "var_cell",
        "mean_var_cell",
    ]
    ks_total = 0

    for stat_type in stats_list:
        if stat_type == "mean_gene":
            axis_along = 0
            stat_test_preds = np.mean(test_preds, axis=axis_along)
            stat_test_labels = np.mean(test_labels, axis=axis_along)
        elif stat_type == "var_gene":
            axis_along = 0
            stat_test_preds = np.var(test_preds, axis=axis_along)
            stat_test_labels = np.var(test_labels, axis=axis_along)
        elif stat_type == "mean_var_gene":
            axis_along = 0
            stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                np.var(test_preds, axis=axis_along) + eps
            )
            stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                np.var(test_labels, axis=axis_along) + eps
            )
        elif stat_type == "mean_cell":
            axis_along = 1
            stat_test_preds = np.mean(test_preds, axis=axis_along)
            stat_test_labels = np.mean(test_labels, axis=axis_along)
        elif stat_type == "var_cell":
            axis_along = 1
            stat_test_preds = np.var(test_preds, axis=axis_along)
            stat_test_labels = np.var(test_labels, axis=axis_along)
        elif stat_type == "mean_var_cell":
            axis_along = 1
            stat_test_preds = np.mean(test_preds, axis=axis_along) / (
                np.var(test_preds, axis=axis_along) + eps
            )
            stat_test_labels = np.mean(test_labels, axis=axis_along) / (
                np.var(test_labels, axis=axis_along) + eps
            )

        stat_test_labels = np.squeeze(stat_test_labels)
        ks_total += ks_2samp(stat_test_preds, stat_test_labels)

    return ks_total


# https://scvae.readthedocs.io/en/latest/ (tuned parameters found under "Training a model")


def objective(trial):
    # Layer size
    # latent_size = trial.suggest_categorical(
    #     "first_layer_size", [32, 64, 128])

    if args.tune_model_type or args.tune_all:
        # model_type = trial.suggest_categorical("model_type", ["VAE", "GMVAE"])
        model_type = trial.suggest_categorical("model_type", ["VAE"])  # debugging
        config["model_type"] = model_type
    if args.tune_reconstruction_distribution or args.tune_all:
        # not including constrained poisson since it apparently can't be sampled from
        # also excluding log_normal 'cause that seems to be causing issues
        # ALSO excluding bernoulli because that needs binarised values
        # it looks like only poisson and gaussian end up working...

        recon_dist = trial.suggest_categorical(
            "reconstruction_distribution",
            [
                "poisson",
                "gaussian",
                # "negative_binomial",
                # "zero_inflated_poisson",
                # "zero_inflated_negative_binomial",
                # "log_normal"
            ],
        )

        config["recon_dist"] = recon_dist

    if args.tune_prior_probabilities_method or args.tune_all:
        if model_type == "GMVAE":
            prior_prob_method = trial.suggest_categorical(
                "prior_probabilities_method", ["uniform", "infer", "learn"]
            )
            config["prior_prob_method"] = prior_prob_method

    if args.tune_latent_size or args.tune_all:
        # arbitrary limits, change based on results
        latent_size = trial.suggest_int("latent_size", 8, args.latent_size_upper_bound)
        config["latent_size"] = latent_size

    if args.tune_hidden_sizes or args.tune_all:
        # arbitrary limits, change based on results
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)

        hidden_sizes = []
        for n in range(num_hidden_layers):
            hidden_sizes.append(
                trial.suggest_int(f"size of hidden layer {n+1}", 50, 300)
            )  # COMPLETELY arbitrary
        config["hidden_sizes"] = hidden_sizes

    if args.tune_number_of_warm_up_epochs or args.tune_all:
        # not sure what proportion of total epochs should be - original paper (Sonderby et. al) uses 10%
        num_warm_up_epochs = trial.suggest_int("number_of_warm_up_epochs", 0, 5)
        config["num_warm_up_epochs"] = num_warm_up_epochs

    if args.tune_kl_weight or args.tune_all:
        print()
        print("tuning kl weight")
        print()
        kl_weight = trial.suggest_float("kl_weight", 0.0, 2.0)
        config["kl_weight"] = kl_weight

    if trial.should_prune():
        raise optuna.TrialPruned()

    # Train model and make simulations of validate data
    test_preds = scVAEEstimator(config)

    # Evaluate
    score = calc_ks(test_preds, valid_labels_npy)

    return score


# Static parameters configuration
if args.stu == "WOT":
    config_file = "./WOTData_exp.json"
    # Change data path based on system
    data_path = (args.data_dir, "WOT", "upto_tp" + str(args.ttp))

    train_file_path = os.path.join(data_path, "train_data.mtx")
    valid_file_path = os.path.join(data_path, "valid_data.mtx")

# elif args.stu == "new_mouse_cell":
#     config_file = "./NewMouseData_exp_new_genes.json"
#     test_file_path = "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Data/new_mouse_cell/w_preprocess_new_genes/test_data.mtx"
#     cell_clusters_file = (
#         "/users/srajakum/scratch/qui2022/new_mouse_cell_clusters_by_tp.yaml"
#     )

# elif args.stu == "zebrafish":

#     config_file = "./ZebrafishData_exp_new_genes.json"
#     valid_file_path = "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Data/zebrafish/w_preprocess_new_genes/valid_data.mtx"
#     cell_clusters_file = "/users/srajakum/scratch/zebrafish_cell_clusters_by_split.yaml"

#     # # debugging - model needs explicit preprocessed values keyword argument for some functionality (data_set.load in train function)
#     # preprocessed_train_values_npy = readMtx(
#     #     "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Data/zebrafish/w_preprocess_new_genes/train_data.mtx").T
#     # preprocessed_train_values = csr_matrix(preprocessed_train_values_npy)
#     # print(
#     #     f'preprocessed train values shape: {preprocessed_train_values.shape}')

#     preprocessed_test_values_npy = readMtx(test_file_path).T
#     preprocessed_test_values = csr_matrix(preprocessed_test_values_npy)
#     print(f"preprocessed test values shape: {preprocessed_test_values.shape}")

#     train_valid_path = "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Data/zebrafish/w_preprocess_new_genes/train_valid.mtx"
#     train_valid_data = csr_matrix(readMtx(train_valid_path).T)
#     print("train_valid shape:", train_valid_data.shape)

## Commented code is zebrafish specific stuff, was trying to incorporate cell clustering

# with open(cell_clusters_file, 'r') as f:
#     # changing to unsafe load because safe load is broken
#     cell_clusters = yaml.unsafe_load(f)

# config = {
#     "config_file": config_file,
#     "number_of_epochs": 50,
#     'cell_clusters': cell_clusters,
#     # changing to include training + validation
#     'train_valid_path': train_valid_path,
#     'train_valid_data': train_valid_data,
#     # 'preprocessed_train_values': train_valid_comb,
#     'preprocessed_test_values': preprocessed_test_values,
#     'sample_size': preprocessed_test_values.shape[0]
# }

elif args.stu == "drosophila":
    config_file = "./DrosophilaData_exp.json"
    data_path = os.path.join(args.data_dir, "drosophila", "upto_tp" + str(args.ttp))


elif args.stu == "mouse_cortex1_chromium":
    config_file = "./MouseCortex1Chromium_exp.json"
    data_path = os.path.join(
        args.data_dir,
        "mouse_cortex",
        "mouse_cortex1_chromium",
        "upto_tp" + str(args.ttp),
    )


elif args.stu == "mouse_cortex2_chromium":
    config_file = "./MouseCortex2Chromium_exp.json"
    data_path = os.path.join(
        args.data_dir,
        "mouse_cortex",
        "mouse_cortex2_chromium",
        "upto_tp" + str(args.ttp),
    )

elif args.stu == "mouse_cortex1_smart_seq":
    config_file = "./MouseCortex1SmartSeq_exp.json"
    data_path = os.path.join(
        args.data_dir,
        "mouse_cortex",
        "mouse_cortex1_smart_seq",
        "upto_tp" + str(args.ttp),
    )


elif args.stu == "mouse_cortex2_smart_seq":
    config_file = "./MouseCortex2SmartSeq_exp.json"
    data_path = os.path.join(
        args.data_dir,
        "mouse_cortex",
        "mouse_cortex2_smart_seq",
        "upto_tp" + str(args.ttp),
    )

elif args.stu == "PBMC":
    config_file = "./PBMCData_exp.json"
    data_path = os.path.join(args.data_dir, "PBMC", "upto_tp" + str(args.ttp))

train_file_path = os.path.join(data_path, "train_data.mtx")
valid_file_path = os.path.join(data_path, "valid_data.mtx")

valid_labels = csr_matrix(readMtx(valid_file_path)).T
valid_labels_npy = np.array(valid_labels.todense())
train_data = csr_matrix(readMtx(train_file_path)).T

with open(os.path.join(data_path, "cell_clusters_dict.yml"), "r") as f:
    cell_clusters_dict = yaml.load(f, Loader=yaml.Loader)

print("valid data shape:", valid_labels.shape)
print("train data shape:", train_data.shape)

# have to provide "preprocessed_train_values" for GMVAE to work,
# otherwise throws error (ELBO for last batch became indefinite)
config = {
    "config_file": config_file,
    "number_of_epochs": 50,
    "preprocessed_train_values": train_data,
    "preprocessed_test_values": valid_labels,
    "sample_size": valid_labels.shape[0],
    "cell_clusters": cell_clusters_dict,
    "ttp": args.ttp,
}


today = date.today()
mdy = today.strftime("%Y-%m-%d")
clock = datetime.now()
hms = clock.strftime("%H-%M-%S")
date_and_time = mdy + "-at-" + hms

start_time = time.time()

# Parameter auto-tune
study = optuna.create_study(direction="minimize")
if args.tuning_time is None:
    study.optimize(objective, n_trials=args.n)
else:
    study.optimize(objective, timeout=args.tuning_time * 60)
print("=" * 70)
best_params = study.best_params
best_trial = study.best_trial
print(
    "Found best params : {} | Best test score = {}".format(
        best_params, best_trial.value
    )
)

path = os.path.join(args.save_dir, date_and_time + "_ttp_" + str(args.ttp))
os.mkdir(path)

with open(os.path.join(path, "tuning_info.txt"), "w") as f:
    f.write("Tuning time: " + str((time.time() - start_time) / 3600) + "\n")
    f.write(f"Number of elapsed trials: {len(study.trials)}" + "\n")
    f.write(f"Number of epochs: " + str(config["number_of_epochs"]) + "\n")
    f.write("Metric type: KS sum" + "\n")
    f.write(f"Best value: {study.best_value}" + "\n")
    f.write(f"Latent size upper bound: {args.latent_size_upper_bound}" + "\n")
    f.write("Best hyperparameters: " + "\n")
    for param, value in study.best_params.items():
        f.write(f"{param}: {value}" + "\n")

with open(os.path.join(path, "hyperparameters.pt"), "wb") as f:
    pickle.dump(study.best_params, f)

param_imp_fig = plot_param_importances(study)
opt_hist_fig = plot_optimization_history(study)
# contour_plot = plot_contour(study, params=["kl_weight", "latent_size"])
param_imp_fig.figure.savefig(os.path.join(path, "param_imp.png"))
opt_hist_fig.figure.savefig(os.path.join(path, "opt_hist.png"))
# contour_plot.figure.savefig(os.path.join(
#     path, "latent_kl_weight_countour.png"))
