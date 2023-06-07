'''
Description:
    Model running for mouse cell data.

Author:
    Jiaqi Zhang
'''
import sys
sys.path.append("../")
sys.path.append("./")
from Py_Utils import readMtx, getTimeStr, addDefaultArg, getDefualtLayerSize
from Running import clusterAugmentation, normalAugmentation

# ------------------------------------------------------------------------------------
# Augmentation for different sampling ratios

def clusterAugmentExp():
    # parameters configuration
    num_genes = 2000
    latent_size = 32
    num_layers = 2
    layer_size_list = getDefualtLayerSize(num_genes, num_layers, latent_size)
    date_str = getTimeStr()
    config = {
        "max_epoch": 50,
        "batch_size": 128,
        "beta": [0.5, 0.5],  # controls relative weight of reconstruction and KL loss, default = 0.5
        "cluster_weight_type": 'sigmoid',  # vanilla (no pi), sigmoid, or softmax
        "layer_size_list": layer_size_list,
        "num_layers": num_layers,
        "learning_rate": 1e-4,
    }
    config = addDefaultArg(config)
    # ------------------------------------------------
    cluster_ls = [0, 3, 18]
    cluster_size = {0: 9094, 3: 1993, 18: 2959}
    for c in cluster_ls:
        print("=" * 70)
        print("Cluster {}".format(c))
        config["data"] = readMtx("../../Data/mouse_cell/w_preprocess/cluster{}_data-{}.mtx".format(c, cluster_size[c]))
        for s in [0.5, 0.25, 0.1, 0.05, 0.03, 0.01]:
            print("*" * 70)
            print("Train data size : {}".format(s))
            config["train_size"] = s
            for t in range(5):
                print("#" * 70)
                print("TRIAL {}".format(t))
                config[
                    "model_save_path"] = "../../Prediction/StructureVAE/mouse_cell_cluster{}-augmented-trial{}-VAE_model-{}.pt".format(
                    c, t, s)
                config[
                    "prediction_save_path"] = "../../Prediction/StructureVAE/mouse_cell_cluster{}-augmented-trial{}-VAE_estimation-{}.npy".format(
                    c, t, s)
                clusterAugmentation(config)

# ------------------------------------------------------------------------------------
# Train, validate, test

def normalAugmentExp():
    # parameters configuration
    num_genes = 2000
    latent_size = 32
    num_layers = 2
    layer_size_list = getDefualtLayerSize(num_genes, num_layers, latent_size)
    date_str = getTimeStr()
    config = {
        "max_epoch" : 50,
        # "max_epoch" : 2,
        "batch_size" : 128,
        "beta" : [0.5, 0.5], # controls relative weight of reconstruction and KL loss, default = 0.5
        "cluster_weight_type" : 'sigmoid',  # vanilla (no pi), sigmoid, or softmax
        "layer_size_list" : layer_size_list,
        "num_layers" : num_layers,
        "learning_rate" : 1e-4,
        # ------------------------------------------
        "train_data": readMtx("../../Data/WOT/w_preprocess/training_all.mtx"),
        "validate_data": readMtx("../../Data/WOT/w_preprocess/validate_all.mtx"),
        "test_data": readMtx("../../Data/WOT/w_preprocess/testing_all.mtx"),
        # ------------------------------------------
        "model_save_path" : "/gpfs/data/rsingh47/jzhan322/Prediction/WOT-VAE_model.pt",
        "prediction_save_path": "/gpfs/data/rsingh47/jzhan322/Prediction/WOT-VAE_estimation.npy",
    }
    config = addDefaultArg(config)
    normalAugmentation(config)




if __name__ == '__main__':
    # clusterAugmentExp()
    normalAugmentExp()
    pass