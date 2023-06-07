"""
Description:
    Train and evaluate the scVAE model.

Authot:
    Jiaqi Zhang
"""
from Py_Utils import readMtx, readH5ad
from cli import *
import umap
import matplotlib.pyplot as plt
import numpy as np
import loompy
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr, spearmanr
import json
from defaults import defaults
import sys
import os
from scipy.sparse import csr_matrix

# from Models.scvae.defaults import defaults
# from Models.scvae.cli import *


def loadArgs(args_filename):
    with open(args_filename) as file:
        args_dict = json.load(file)
    return args_dict


def train_model(args_dict):
    (model, output_sets) = train(**args_dict)
    return (model, output_sets)


def eval_model(args_dict, model, n_cells_new):
    (
        test_preds,
        transformed_evaluation_set,
        reconstructed_evaluation_set,
        _,
    ) = model.evaluate(
        **args_dict,
        # model = model,
        sample_size=n_cells_new,
    )
    return test_preds, transformed_evaluation_set, reconstructed_evaluation_set


def train_and_predict(args_dict):
    output_sets = train(
        **args_dict,
        latent_size=args_dict["latent_size"],
        kl_weight=args_dict["kl_weight"],
        number_of_epochs=args_dict["number_of_epochs"],
    )
    transformed_evaluation_set = output_sets[0].values
    reconstructed_evaluation_set = output_sets[1].values
    return transformed_evaluation_set, reconstructed_evaluation_set


# -------------------------------------------------------------------------


def clusterAugmentation(config):
    ...


def normalAugmentation(args_dict, train_file_path, test_file_path, cell_clusters):

    test_labels = csr_matrix(readMtx(test_file_path)).T
    test_labels_npy = np.array(test_labels.todense())
    train_data = csr_matrix(readMtx(train_file_path)).T

    args_dict["data_set_file_or_name"] = train_file_path

    # specify preprocessed_triain_values
    args_dict["preprocessed_values"] = train_data

    # CHANGE: also specify values in arg dict?
    args_dict["values"] = train_data

    # setting labels as cell clusters
    args_dict["labels"] = cell_clusters["train_clusters"]

    # Train the model
    model = train(**args_dict)

    # evaluate
    args_dict["sample_size"] = test_labels.shape[0]

    # specify preprocessed values here as well

    args_dict["preprocessed_values"] = test_labels

    args_dict["values"] = test_labels

    args_dict["data_set_file_or_name"] = test_file_path
    args_dict["labels"] = cell_clusters["test_clusters"]

    sampled_data, _, _, _ = evaluate(model=model, **args_dict)

    # print("Dense train data size = {}".format(
    #     sys.getsizeof(transformed_evaluation_set)))
    # print("Dense predictions data size = {}".format(
    #     sys.getsizeof(reconstructed_evaluation_set)))

    # print("Sparse train data size = {}".format(
    #     sys.getsizeof(transformed_evaluation_set)))
    # print("Sparse predictions data size = {}".format(
    #     sys.getsizeof(reconstructed_evaluation_set)))

    # print(
    #     f'transormed train data shape: {transformed_evaluation_set.shape}')
    # print(f'test labels shape: {test_labels.shape}')
    # print(f'simulated data shape: {reconstructed_evaluation_set.shape}')

    # num_cells_test = test_labels.shape[0]
    # num_cells_sim = reconstructed_evaluation_set.shape[0]
    # test_ind = np.random.choice(num_cells_sim, size=num_cells_test)
    # test_preds = reconstructed_evaluation_set[test_ind, :]
    # print(f'test preds shape: {test_preds.shape}')

    # test_preds = model.sample(n_cells_new)[0]

    # mean_gene_exp_pred = np.mean(test_pred, axis=0)
    # mean_gene_exp_labels = np.mean(test_data, axis=0)
    # test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
    # test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
    # print('Test PCC:', test_pcc)
    # print('Test SCC:', test_scc)

    # pred_cell_avg = np.mean(test_preds, axis=1)
    # label_cell_avg = np.mean(test_data, axis=1)
    # cell_ks_stat = ks_2samp(pred_cell_avg, label_cell_avg).statistic
    # pred_gene_avg = np.mean(test_preds, axis=0)
    # label_gene_avg = np.mean(test_labels, axis=0)
    # gene_ks_stat = ks_2samp(mean_gene_exp_pred, mean_gene_exp_labels).statistic
    # test_ks = (cell_ks_stat + gene_ks_stat) / 2
    # print('Test KS:', test_ks)

    # summary = {
    #     "test_predictions": coo_matrix(reconstructed_evaluation_set),
    #     "test_labels": coo_matrix(transformed_evaluation_set),
    # }

    # test_labels = coo_matrix(test_labels)

    # test_preds = coo_matrix(sampled_data)

    # mean_cell_test_labels_sorted = np.sort(np.mean(test_labels, axis=1))
    # mean_gene_test_labels_sorted = np.sort(np.mean(test_labels, axis=0))
    # var_cell_test_labels_sorted = np.sort(np.var(test_labels, axis=1))
    # var_gene_test_labels_sorted = np.sort(np.var(test_labels, axis=0))

    # mean_cell_sampled_data_sorted = np.sort(np.mean(sampled_data, axis=1))
    # mean_gene_sampled_data_sorted = np.sort(np.mean(sampled_data, axis=0))
    # var_cell_sampled_data_sorted = np.sort(np.var(sampled_data, axis=1))
    # var_gene_sampled_data_sorted = np.sort(np.var(sampled_data, axis=0))

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots(2, 2)

    # y_cell = np.arange(0, 1, 1 / test_labels.shape[0])
    # y_gene = np.arange(0, 1, 1 / test_labels.shape[1])

    # ax[0, 0].plot(mean_cell_test_labels_sorted, y_cell, label='True')
    # ax[0, 0].plot(mean_cell_sampled_data_sorted, y_cell, label='Estimated')
    # ax[0, 0].set_title('Mean Cell CDFs')

    # ax[0, 1].plot(mean_gene_test_labels_sorted, y_gene, label='True')
    # ax[0, 1].plot(mean_gene_sampled_data_sorted, y_gene, label='Estimated')
    # ax[0, 1].set_title('Mean Gene CDFs')

    # ax[1, 0].plot(var_cell_test_labels_sorted, y_cell, label='True')
    # ax[1, 0].plot(var_cell_sampled_data_sorted, y_cell, label='Estimated')
    # ax[1, 0].set_title('Var Cell CDFs')

    # ax[1, 1].plot(var_gene_test_labels_sorted, y_gene, label='True')
    # ax[1, 1].plot(var_gene_sampled_data_sorted, y_gene, label='Estimated')
    # ax[1, 1].set_title('Var Gene CDFs')

    # fig.savefig(
    #     "/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/new_baseline_results/scvae/zebrafish/marginal_cdfs.png")

    # np.savez_compressed(config["prediction_save_path"], **summary)
    # mmwrite("../../Prediction/scVAE/PBMC-scVAE_estimation.mtx", reconstructed_evaluation_set)
    # mmwrite("../../Prediction/scVAE/PBMC-scVAE_train_data.mtx", transformed_evaluation_set)

    # mmwrite(config["estimation_filename"], reconstructed_evaluation_set)
    # mmwrite(config["train_data_filename"], transformed_evaluation_set)
    # mmwrite(config["test_labels_filename"], test_labels)

    # with open(config["test_stats_filename"], "w") as f:
    #     f.write(
    #         "Test statistics computed between test data random subset of simulated data produced from training")
    #     f.write("Test PCC:", test_pcc)
    #     f.write("Test SCC:", test_scc)
    #     f.write("Test KS:", test_ks)

    return sampled_data


# -------------------------------------------------------------------------

## Commented code below was attempted implemenation using cell clusters

# def modelTrainForCV(config):
#     config_file = config["config_file"]
#     args_dict = loadArgs(config_file)
#     # Set parameters
#     args_dict["model_type"] = config["model_type"] if "model_type" in config else None
#     args_dict["latent_size"] = config["latent_size"] if "latent_size" in config else None
#     args_dict["kl_weight"] = config["kl_weight"] if "kl_weight" in config else None
#     args_dict["reconstruction_distribution"] = config["recon_dist"] if "recon_dist" in config else None
#     args_dict["prior_probabilities_method"] = config["prior_prob_method"] if "prior_prob_method" in config else None
#     args_dict["hidden_sizes"] = config["hidden_sizes"] if "hidden_sizes" in config else None
#     args_dict["number_of_warm_up_epochs"] = config["num_warm_up_epochs"] if "num_warm_up_epochs" in config else None
#     args_dict["number_of_epochs"] = config["number_of_epochs"] if "number_of_epochs" in config else 50

#     print(f'model type: {args_dict["model_type"]}')

#     # get saved cell cluster info

#     cell_clusters = config['cell_clusters']

#     # replacing with train + valid labels for debugging
#     # args_dict['labels'] = np.array(cell_clusters['train'])

#     args_dict['labels'] = np.array(cell_clusters['train_valid'])

#     # get preprocessed values (debugging)
#     # IMPORTANT: changing to input train + valid

#     # args_dict['preprocessed_values'] = config['preprocessed_train_values']
#     args_dict['preprocessed_values'] = config['train_valid_data']
#     args_dict['values'] = config['train_valid_data']

#     # Train the model

#     # IMPORTANT: including data split because giving training + valid data
#     # this requires: replacing train data file name, setting 'preprocessed_values' to train + valid data (above)
#     args_dict['data_set_file_or_name'] = config['train_valid_path']
#     args_dict['split_data_set'] = True
#     args_dict['example_names'] = args_dict['labels']
#     model = train(**args_dict)

#     # evaluate
#     print('in Running.py, before evaluate:')
#     print('preprocessed test values dim:',
#           config['preprocessed_test_values'].shape)
#     print('unique test labels:', set(cell_clusters['test']))

#     args_dict['sample_size'] = config['sample_size']
#     args_dict['preprocessed_values'] = config['preprocessed_test_values']
#     args_dict['labels'] = np.array(cell_clusters['test'])
#     args_dict['data_set_file_or_name'] = args_dict['evaluation_data_set_name']

#     print('cross-checking with args dict:')
#     print('preprocessed test values dim:',
#           args_dict['preprocessed_values'].shape)
#     print('unique test labels:', set(args_dict['labels']))
#     print('data set file name:', args_dict['data_set_file_or_name'])

#     sampled_data, _, _, _ = evaluate(
#         model=model,
#         **args_dict,
#     )

#     print(f'sampled data shape: {sampled_data.shape}')
#     assert(sampled_data.shape ==
#            config['preprocessed_test_values'].shape), "labels and predictions not same shape"

#     return sampled_data


def modelTrainForCV(config):
    config_file = config["config_file"]
    args_dict = loadArgs(config_file)
    # Set parameters
    args_dict["model_type"] = config["model_type"] if "model_type" in config else None
    args_dict["latent_size"] = (
        config["latent_size"] if "latent_size" in config else None
    )
    args_dict["kl_weight"] = config["kl_weight"] if "kl_weight" in config else None
    args_dict["reconstruction_distribution"] = (
        config["recon_dist"] if "recon_dist" in config else None
    )
    args_dict["prior_probabilities_method"] = (
        config["prior_prob_method"] if "prior_prob_method" in config else None
    )
    args_dict["hidden_sizes"] = (
        config["hidden_sizes"] if "hidden_sizes" in config else None
    )
    args_dict["number_of_warm_up_epochs"] = (
        config["num_warm_up_epochs"] if "num_warm_up_epochs" in config else None
    )
    args_dict["number_of_epochs"] = (
        config["number_of_epochs"] if "number_of_epochs" in config else 50
    )

    print("kl weight in args dict:", args_dict["kl_weight"])

    print(f'model type: {args_dict["model_type"]}')

    args_dict["data_set_file_or_name"] = os.path.join(
        args_dict["data_directory"], "upto_tp" + str(config["ttp"]), "train_data.mtx"
    )

    print("data set file name:", args_dict["data_set_file_or_name"])

    # specify preprocessed_triain_values
    args_dict["preprocessed_values"] = config["preprocessed_train_values"]
    print("preprocessed values shape:", args_dict["preprocessed_values"].shape)

    # CHANGE: also specify values in arg dict?
    args_dict["values"] = config["preprocessed_train_values"]

    # setting labels as cell clusters
    args_dict["labels"] = config["cell_clusters"]["train_clusters"]

    # Train the model
    model = train(**args_dict)

    # evaluate
    args_dict["sample_size"] = config["sample_size"]

    # specify preprocessed values here as well

    args_dict["preprocessed_values"] = config["preprocessed_test_values"]

    args_dict["values"] = config["preprocessed_test_values"]

    args_dict["data_set_file_or_name"] = os.path.join(
        args_dict["data_directory"], "upto_tp" + str(config["ttp"]), "test_data.mtx"
    )
    args_dict["labels"] = config["cell_clusters"]["test_clusters"]

    sampled_data, _, _, _ = evaluate(model=model, **args_dict)

    return sampled_data


if __name__ == "__main__":
    args_dict = loadArgs("./splat_simulation_exp.json")
    # model = train_model(args_dict)
    (
        transformed_evaluation_set,
        reconstructed_evaluation_set,
        latent_evaluation_sets,
    ) = train_model(**args_dict)
    # ------------------------------------------------------
    generated_data_values = reconstructed_evaluation_set
    values = transformed_evaluation_set

    mean_gene_exp_pred = np.mean(generated_data_values, axis=0)
    mean_gene_exp_labels = np.mean(values, axis=0)
    test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
    test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
    print("Test PCC:", test_pcc)
    print("Test SCC:", test_scc)

    test_pcc_str = str(np.round(test_pcc, 3))
    test_scc_str = str(np.round(test_scc, 3))
    cc_str = f"Test PCC = {test_pcc_str}\n" f"Test SCC = {test_scc_str}"

    scores_and_labels = np.concatenate((values, generated_data_values), axis=0)
    # model = umap.UMAP().fit(values)
    model = umap.UMAP().fit(scores_and_labels)
    emedding = model.transform(scores_and_labels)
    plt.scatter(
        emedding[:1500, 0],
        emedding[:1500, 1],
        color="blue",
        alpha=0.5,
        s=5,
        label="True",
    )
    plt.scatter(
        emedding[1500:, 0],
        emedding[1500:, 1],
        color="orange",
        alpha=0.5,
        s=5,
        label="Simulated",
    )
    plt.legend()
    plt.title("UMAP", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Dimension 1", fontsize=15)
    plt.ylabel("Dimension 2", fontsize=15)
    plt.show()


# model = arguments.func(**vars(arguments))
# generated_data = model.sample(sample_size=1500)[0]
# generated_data_values = generated_data.values
#
# # # Save prediction
# # with loompy.connect("./gene_by_cell_testing_all.loom") as data_file:
# #     values = data_file[:, :].T
#
# # ------------------------------------------------------
# mean_gene_exp_pred = np.mean(generated_data_values, axis=0)
# mean_gene_exp_labels = np.mean(values, axis=0)
# test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
# test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
# print('Test PCC:', test_pcc)
# print('Test SCC:', test_scc)
#
# test_pcc_str = str(np.round(test_pcc, 3))
# test_scc_str = str(np.round(test_scc, 3))
# cc_str = (
#     f"Test PCC = {test_pcc_str}\n"
#     f"Test SCC = {test_scc_str}"
# )
#
# scores_and_labels = np.concatenate((values, generated_data_values), axis=0)
# # model = umap.UMAP().fit(values)
# model = umap.UMAP().fit(scores_and_labels)
# emedding = model.transform(scores_and_labels)
# plt.scatter(emedding[:1500, 0], emedding[:1500, 1], color='blue', alpha=0.5, s=5, label='True')
# plt.scatter(emedding[1500:, 0], emedding[1500:, 1], color='orange', alpha=0.5, s=5, label='Simulated')
# plt.legend()
# plt.title('UMAP', fontsize=15)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('Dimension 1', fontsize=15)
# plt.ylabel('Dimension 2', fontsize=15)
# plt.show()
