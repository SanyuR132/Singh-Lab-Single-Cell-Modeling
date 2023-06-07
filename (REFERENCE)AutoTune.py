'''
Description:
    Automatically tune model parameters.

Author:
    Jiaqi Zhang
'''
import optuna
from scipy.stats import ks_2samp, pearsonr
import numpy as np
import scanpy
import joblib

import sys
import os
import shutil
sys.path.append("../")
from Models.Py_Utils import readMtx, getTimeStr, addDefaultArg, getDefualtLayerSize, prepareDataset, readH5ad

# ----------------------------------------------

def tuneDCA():
    # https://github.com/theislab/dca
    from Models.dca.Running import modelTrainForCV as DCAEstimator

    def validateLoss(trial):
        # Layer size
        layer_size = trial.suggest_categorical("first_layer_size", [32, 64, 128])
        config["layer_size"] = [layer_size, layer_size // 2, layer_size]
        # Hidden dropout
        dropout = trial.suggest_float("dropout", 0.0, 1.0)
        config["dropout"] = dropout
        # Train model and make simulations of validate data
        validate_preds, validate_labels = DCAEstimator(config)
        # Evaluate
        if args.s == "KS":
            pred_cell_avg = np.mean(validate_preds, axis=1)
            label_cell_avg = np.mean(validate_labels, axis=1)
            cell_ks_stat = ks_2samp(pred_cell_avg, label_cell_avg).statistic
            pred_gene_avg = np.mean(validate_preds, axis=0)
            label_gene_avg = np.mean(validate_labels, axis=0)
            gene_ks_stat = ks_2samp(pred_gene_avg, label_gene_avg).statistic
            score = (cell_ks_stat + gene_ks_stat) / 2
        elif args.s == "PCC":
            pred_cell_avg = np.mean(validate_preds, axis=1)
            label_cell_avg = np.mean(validate_labels, axis=1)
            cell_pcc = pearsonr(pred_cell_avg, label_cell_avg)[0]
            pred_gene_avg = np.mean(validate_preds, axis=0)
            label_gene_avg = np.mean(validate_labels, axis=0)
            gene_pcc = pearsonr(pred_gene_avg, label_gene_avg)[0]
            score = 1 - (cell_pcc + gene_pcc) / 2
        else:
            raise ValueError("Unknown evaluation score {}!".format(args.s))
        return score


    # Static parameters configuration
    config = {
        "train_data": "../Data/mouse_cell/wo_preprocess/training_all.mtx",
        "validate_data": "../Data/mouse_cell/wo_preprocess/validate_all.mtx",
    }
    # Prepare datasets
    print("=" * 70)
    print("START LOADING DATA...")
    if config["train_data"].split(".")[-1] == "mtx":
        train_data = readMtx(config["train_data"])  # cell by gene
        train_data = scanpy.AnnData(train_data)
        validate_data = readMtx(config["validate_data"])  # cell by gene
        validate_data = scanpy.AnnData(validate_data)
    elif config["train_data"].split(".")[-1] == "h5ad":
        train_data = readH5ad(config["train_data"])  # cell by gene
        train_data = scanpy.AnnData(train_data)
        validate_data = readH5ad(config["validate_data"])  # cell by gene
        validate_data = scanpy.AnnData(validate_data)
    else:
        raise ValueError("Files of {} format cannot be processed!".format(config["train_data"].split(".")[-1]))
    filters = scanpy.pp.filter_genes(train_data, min_counts=1, inplace=False)
    filtered_train_data = train_data[:, filters[0]]  # cells x genes
    filtered_validate_data = validate_data[:, filters[0]]  # cells x genes

    config["train_data"], config["validate_data"] = filtered_train_data, filtered_validate_data
    print("[ Training ] Num of cells = {} | Num of genes = {} ".format(filtered_train_data.shape[0], filtered_train_data.shape[1]))
    print("[ Validate ] Num of cells = {} | Num of genes = {} ".format(filtered_validate_data.shape[0], filtered_validate_data.shape[1]))

    # Parameter auto-tune
    study = optuna.create_study()
    study.optimize(validateLoss, n_trials=args.n)
    print("=" * 70)
    best_params = study.best_params
    best_trial = study.best_trial
    print("Found best paras : {} | Best validate loss = {}".format(best_params, best_trial.value))


def tuneSCVAE():
    # https://scvae.readthedocs.io/en/latest/
    from Models.scvae.Running import modelTrainForCV as scVAEEstimator

    def validateLoss(trial):
        # Layer size
        latent_size = trial.suggest_categorical("first_layer_size", [32, 64, 128])
        config["latent_size"] = latent_size
        # Kl weight
        kl_weight = trial.suggest_float("kl_weight", 0.0, 1.0)
        config["kl_weight"] = kl_weight
        # Train model and make simulations of validate data
        validate_preds, validate_labels = scVAEEstimator(config)
        # Evaluate
        if args.s == "KS":
            pred_cell_avg = np.mean(validate_preds, axis=1)
            label_cell_avg = np.mean(validate_labels, axis=1)
            cell_ks_stat = ks_2samp(pred_cell_avg, label_cell_avg).statistic
            pred_gene_avg = np.mean(validate_preds, axis=0)
            label_gene_avg = np.mean(validate_labels, axis=0)
            gene_ks_stat = ks_2samp(pred_gene_avg, label_gene_avg).statistic
            score = (cell_ks_stat + gene_ks_stat) / 2
        elif args.s == "PCC":
            pred_cell_avg = np.mean(validate_preds, axis=1)
            label_cell_avg = np.mean(validate_labels, axis=1)
            cell_pcc = pearsonr(pred_cell_avg, label_cell_avg)[0]
            pred_gene_avg = np.mean(validate_preds, axis=0)
            label_gene_avg = np.mean(validate_labels, axis=0)
            gene_pcc = pearsonr(pred_gene_avg, label_gene_avg)[0]
            score = 1 - (cell_pcc + gene_pcc) / 2
        else:
            raise ValueError("Unknown evaluation score {}!".format(args.s))
        return score


    # Static parameters configuration
    config = {
        "config_file": "./MouseData_exp.json",
        "number_of_epochs": 5,
    }
    # Parameter auto-tune
    study = optuna.create_study()
    study.optimize(validateLoss, n_trials=args.n)
    print("=" * 70)
    best_params = study.best_params
    best_trial = study.best_trial
    print("Found best paras : {} | Best validate loss = {}".format(best_params, best_trial.value))



if __name__ == '__main__':
    import argparse
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('-m', default="scVAE", type=str, help="The model name, {DCA, scVAE}.")
    main_parser.add_argument('-s', default="KS", type=str, help="The type of score function, {KS, PCC}.")
    main_parser.add_argument('-n', default=5, type=int, help="The number of trials.")
    args = main_parser.parse_args()

    # -----
    if args.m == "DCA":
        tuneDCA()
    elif args.m == "scVAE":
        tuneSCVAE()
    else:
        raise ValueError("Unimplemented model {}!".format(args.m))