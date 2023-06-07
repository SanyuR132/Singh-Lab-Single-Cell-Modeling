import optuna
from scipy.stats import ks_2samp
import numpy as np
import joblib

# import sys
# sys.path.append("../")
# from Models.Py_Utils import readMtx, addDefaultArg, getDefualtLayerSize, prepareDataset

# ----------------------------------------------

def tuneStructureModel():
    # This line imports the old structure model, you may need to use new codes
    from Models.StructureModel.Running import modelTrainForCV as VAEEstimator


    # ==================================================
    # Score function

    def validateLoss(trial):
        # Specify search space for parameters.
        # You can define a search range with `suggest_float` or a set of values with `suggest_categorical`.
        # Details may refer to https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
        num_layers = 2
        latent_size = trial.suggest_categorical("latent_size", [32, 64, 128])
        layer_size_list = getDefualtLayerSize(config["num_genes"], num_layers, latent_size)
        config["layer_size_list"] = layer_size_list
        beta = trial.suggest_float("beta", 0.0, 1.0)
        config["beta"] = [beta, 1 - beta]  # controls relative weight of reconstruction and KL loss, default = 0.5
        # -----
        # I organized the old structure model into a function and use `config` to specify model parameters.
        # How to run the model depends on your codes.
        validate_preds, validate_labels = VAEEstimator(config) # Train model and make simulations of validate data
        # -----
        # Evaluate
        pred_cell_avg = np.mean(validate_preds, axis=1)
        label_cell_avg = np.mean(validate_labels, axis=1)
        cell_ks_stat = ks_2samp(pred_cell_avg, label_cell_avg).statistic
        pred_gene_avg = np.mean(validate_preds, axis=0)
        label_gene_avg = np.mean(validate_labels, axis=0)
        gene_ks_stat = ks_2samp(pred_gene_avg, label_gene_avg).statistic
        return cell_ks_stat + gene_ks_stat

    # ==================================================

    # ==================================================
    # This part inits configurations and loads data.
    # You should replace this part with your codes

    config = {
        "max_epoch": 2,
        "batch_size": 128,
        "cluster_weight_type": 'sigmoid',  # vanilla (no pi), sigmoid, or softmax
        # "layer_size_list": layer_size_list,
        "learning_rate": 1e-4,
        # ------------------------------------------
        "train_data": readMtx("../Data/mouse_cell/w_preprocess/training_all.mtx"),
        "validate_data": readMtx("../Data/mouse_cell/w_preprocess/validate_all.mtx"),
        "test_data": readMtx("../Data/mouse_cell/w_preprocess/testing_all.mtx"),
    }
    config = addDefaultArg(config)
    # Load data
    print("=" * 70)
    print("START LOADING DATA...")
    train_set, valid_set, test_set = prepareDataset(
        config["train_data"], config["validate_data"], config["test_data"], config["batch_size"]
    )
    config["train_data"], config["validate_data"], config["test_data"] = train_set, valid_set, test_set
    # Num of genes
    config["num_genes"] = train_set.shape[1]
    # ==================================================

    # ==================================================
    # Parameter auto-tuning with optuna
    study = optuna.create_study()
    study.optimize(validateLoss, n_trials=5)
    print("=" * 70)
    best_params = study.best_params
    best_trial = study.best_trial
    print("Found best paras : {} | Best score = {}".format(best_params, best_trial.value))
    joblib.dump(study, "./optuna_study.pkl")
    # ==================================================



if __name__ == '__main__':
    # The `optuna` package is required; you can install the `joblib` package if you want to save the parameter tuning object (line 81).
    # The parameter tuning can be parallel, you can check https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#
    # The package also offers several visualization of optimization history (https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html)
    tuneStructureModel()