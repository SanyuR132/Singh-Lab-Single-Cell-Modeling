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

def tunePRESCIENT():
    from Models.PRESCIENT.Running import modelTrainForCV as PRESCIENTEstimator
    import argparse
    import sklearn
    from sklearn.decomposition import PCA

    def create_train_parser(
        k_dim = None, tau=None, clip=None,
        train_epochs=None, train_lr = None, train_batch = None, train_sd = None,
        layers = None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true')
        parser.add_argument('--gpu', default=7, type=int,
                            help="Designate GPU number as an integer (compatible with CUDA).")
        parser.add_argument('--out_dir', default='../Prediction/PRESCIENT/',
                            help="Directory for storing training output.")
        parser.add_argument('--seed', type=int, default=2, help="Set seed for training process.")
        # -- data options
        parser.add_argument('-i', '--data_path', default="../Data/mouse_cell/pt_data/before_day{}_data.pt".format(day),
                            help="Input PRESCIENT data torch file.")
        # parser.add_argument("-i", "--data_path", default="/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/mouse_cell/pt_data/before_day{}_data.pt".format(day),
        #                     required=False, help="Path to PRESCIENT data file stored as a torch pt.")
        # parser.add_argument("-i", "--data_path",
        #                     default="/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/new_mouse_cell/pt_data/before_day{}_data.pt".format(day),
        #                     required=False, help="Path to PRESCIENT data file stored as a torch pt.")
        parser.add_argument('--weight_name', default="kegg-growth",
                            help="Designate descriptive name of growth parameters for filename.")
        # -- model options
        parser.add_argument('--loss', default='euclidean', help="Designate distance function for loss.")
        parser.add_argument('--k_dim', default=500 if k_dim is None else k_dim, type=int, help="Designate hidden units of NN.")
        parser.add_argument('--activation', default='softplus', help="Designate activation function for layers of NN.")
        parser.add_argument('--layers', default=1 if layers is None else layers, type=int,
                            help="Choose number of layers for neural network parameterizing the potential function.")
        # -- pretrain options
        parser.add_argument('--pretrain_epochs', default=5, type=int,
                            help="Number of epochs for pretraining with contrastive divergence.")
        # -- train options
        parser.add_argument('--train_epochs', default=200 if train_epochs is None else train_epochs, type=int, help="Number of epochs for training.")
        parser.add_argument('--train_lr', default=1e-4 if train_lr is None else train_lr, type=float,
                            help="Learning rate for Adam optimizer during training.")
        parser.add_argument('--train_dt', default=0.1, type=float, help="Timestep for simulations during training.")
        parser.add_argument('--train_sd', default=0.5 if train_sd is None else train_sd, type=float,
                            help="Standard deviation of Gaussian noise for simulation steps.")
        parser.add_argument('--train_tau', default=1e-6 if tau is None else tau, type=float, help="Tau hyperparameter of PRESCIENT.")
        parser.add_argument('--train_batch', default=0.1 if train_batch is None else train_batch, type=float, help="Batch size for training.")
        parser.add_argument('--train_clip', default=0.25 if clip is None else clip, type=float, help="Gradient clipping threshold for training.")
        parser.add_argument('--save', default=100, type=int, help="Save model every n epochs.")
        # -- run options
        parser.add_argument('--pretrain', type=bool, default=True, help="If True, pretraining will run.")
        parser.add_argument('--train', type=bool, default=True,
                            help="If True, training will run with existing pretraining torch file.")
        parser.add_argument('--config')

        parser.add_argument('-m')
        parser.add_argument('-s')
        parser.add_argument('-d')
        parser.add_argument('-n')
        return parser

    # =====================================================

    def create_simulate_parser(num_cells=None, k_dim=None, tau=None, lr=None, clip=None, sd=None, batch=None, layers=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--data_path", default="../Data/mouse_cell/pt_data/before_day{}_data.pt".format(day), required=False,
                            help="Path to PRESCIENT data file stored as a torch pt.")
        # parser.add_argument("-i", "--data_path", default="/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/mouse_cell/pt_data/before_day{}_data.pt".format(day),
        #                     required=False, help="Path to PRESCIENT data file stored as a torch pt.")
        # parser.add_argument("-i", "--data_path",
        #                     default="/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/new_mouse_cell/pt_data/before_day{}_data.pt".format(day),
        #                     required=False, help="Path to PRESCIENT data file stored as a torch pt.")
        parser.add_argument("--model_path", default="../Prediction/PRESCIENT/kegg-growth-softplus_{}_{}-{}-{}-{}-{}-{}-{}-day{}/".format(
            layers, k_dim, tau, sd, lr, clip, batch, args.s, day),
                            required=False, help="Path to directory containing PRESCIENT model for simulation.")
        parser.add_argument("--seed", default=2, required=False,
                            help="Choose the seed of the trained model to use for simulations.")
        parser.add_argument("--epoch", type=str, required=False,
                            help="Choose which epoch of the model to use for simulations.")
        parser.add_argument("--num_sims", default=1, help="Number of simulations to run.")
        parser.add_argument("--num_cells", default=validate_data_num if num_cells is None else num_cells, help="Number of cells per simulation.")
        parser.add_argument("--num_steps", default=1, required=False,
                            help="Define number of forward steps of size dt to take.")
        parser.add_argument("--gpu", default=None, required=False, help="If available, assign GPU device number.")
        parser.add_argument("--celltype_subset", default=None, required=False,
                            help="Randomly sample initial cells from a particular celltype defined in metadata.")
        parser.add_argument("--tp_subset", default=None, required=False,
                            help="Randomly sample initial cells from a particular timepoint.")
        parser.add_argument("-o", "--out_path", required=False, default="../../Prediction/PRESCIENT/",
                            help="Path to output directory.")

        parser.add_argument('-m')
        parser.add_argument('-s')
        parser.add_argument('-d')
        parser.add_argument('-n')
        return parser

    # =====================================================

    def reverseDimension(ref_data, data):
        scaler = sklearn.preprocessing.StandardScaler()
        standard_true_train_data = scaler.fit_transform(ref_data)
        pca = PCA(n_components=50)
        pca.fit_transform(standard_true_train_data)
        # -----
        data = scaler.inverse_transform(pca.inverse_transform(data))
        return data


    # =====================================================

    def validateLoss(trial):
        # Latent dimension
        k_dim = trial.suggest_categorical("k_dim", [50, 100, 150, 200])
        # tau
        tau = trial.suggest_float("tau", 0.0, 1.0)
        # Gradient clip
        clip = trial.suggest_categorical("clip", [0.25, 0.5, 0.75])
        # Learning rate
        lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 1e-1])
        # lr = None
        # Train batch
        # train_batch_ratio = trial.suggest_categorical("batch_ratio", [0.1, 0.2, 0.3, 0.4, 0.5])
        train_batch_ratio = 0.1
        # Train SD
        train_sd = trial.suggest_categorical("SD", np.arange(0.0, 1.1, 0.1))
        # train_sd = 0.5
        # Layers
        layers = trial.suggest_categorical("layers", [1, 2, 3])
        # layers = None
        # -----
        train_parser = create_train_parser(k_dim, tau, clip, train_epochs, lr, train_batch_ratio, train_sd, layers)
        train_args = train_parser.parse_args()
        for i in ["m", "s", "d", "n"]:
            train_args.__delattr__(i)
        # -----
        simulation_parser = create_simulate_parser(k_dim=k_dim, tau=tau, lr=lr, clip=clip, sd=train_sd, batch=train_batch_ratio, layers=layers)
        simulate_args = simulation_parser.parse_args()
        for i in ["m", "s", "d", "n"]:
            simulate_args.__delattr__(i)
        # Train model and make simulations of validate data
        validate_preds, validate_labels = PRESCIENTEstimator(train_args, simulate_args, metric=args.s, day=day)
        # Reverse to original spaces
        reversed_pred = reverseDimension(original_validate_data, validate_preds) #TODO: reverse到哪个空间？train？validate？
        # Concatenate back to train data and predict the next time step

        # Evaluate
        if args.s == "KS":
            pred_cell_avg = np.mean(reversed_pred, axis=1)
            label_cell_avg = np.mean(original_validate_data, axis=1)
            cell_ks_stat = ks_2samp(pred_cell_avg, label_cell_avg).statistic
            pred_gene_avg = np.mean(reversed_pred, axis=0)
            label_gene_avg = np.mean(original_validate_data, axis=0)
            gene_ks_stat = ks_2samp(pred_gene_avg, label_gene_avg).statistic
            score = (cell_ks_stat + gene_ks_stat) / 2
        elif args.s == "PCC":
            pred_cell_avg = np.mean(reversed_pred, axis=1)
            label_cell_avg = np.mean(original_validate_data[np.random.choice(np.arange(original_validate_data.shape[0]), size=reversed_pred.shape[0])], axis=1)
            cell_pcc = pearsonr(pred_cell_avg, label_cell_avg)[0]
            pred_gene_avg = np.mean(reversed_pred, axis=0)
            label_gene_avg = np.mean(original_validate_data, axis=0)
            gene_pcc = pearsonr(pred_gene_avg, label_gene_avg)[0]
            score = 1 - (cell_pcc + gene_pcc) / 2
        else:
            raise ValueError("Unknown evaluation score {}!".format(args.s))
        # # Delete records
        # dir_list = [each for each in os.listdir("../Prediction/PRESCIENT/") if "kegg" in each]
        # for each in dir_list:
        #     shutil.rmtree("../Prediction/PRESCIENT/{}".format(each))
        return score

    # =====================================================

    day = "11" if args.d is None else args.d
    train_epochs = 5 #TODO: num of steps
    n_trials = args.n
    print("*" * 70)
    print("Total num of trials = {} | Total num of epochs = {}".format(n_trials, train_epochs))
    print("*" * 70)

    original_validate_data = readMtx("../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_{}_5.mtx".format(int(day) + 1))
    # original_validate_data = readMtx("/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_{}_5.mtx".format(int(day) + 1))
    # original_validate_data = readMtx("/gpfs/data/rsingh47/jzhan322/Structure_VAE/Data/new_mouse_cell/gene_exp_mat_time_{}_5_rc.mtx".format(int(day) + 1))
    validate_data_num = original_validate_data.shape[0]

    # Parameter auto-tune
    study = optuna.create_study()
    study.optimize(validateLoss, n_trials=n_trials)
    print("=" * 70)
    best_params = study.best_params
    best_trial = study.best_trial
    print("Found best paras : {} | Best validate loss = {}".format(best_params, best_trial.value))
    # Save study
    joblib.dump(study, "./PRESCIENT-day{}_study.pkl".format(day))



if __name__ == '__main__':
    import argparse
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('-m', default="PRESCIENT", type=str, help="The model name, {PRESCIENT, structure, DCA, scVAE}.")
    main_parser.add_argument('-s', default="KS", type=str, help="The type of score function, {KS, PCC}.")
    main_parser.add_argument('-d', default=None, type=str, help="The target day.")
    main_parser.add_argument('-n', default=5, type=int, help="The number of trials.")
    args = main_parser.parse_args()
    # -----
    tunePRESCIENT()