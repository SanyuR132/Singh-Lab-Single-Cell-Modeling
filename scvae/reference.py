'''
Description:
    Train and evaluate the scVAE model.

Authot:
    Jiaqi Zhang
'''

import sys
sys.path.append("./")

from defaults import defaults
from cli import *
# from Models.scvae.defaults import defaults
# from Models.scvae.cli import *
import json

from scipy.stats import pearsonr, spearmanr
import loompy
import numpy as np
import matplotlib.pyplot as plt
import umap


def _parse_default(default):
    if not isinstance(default, bool) and default != 0 and not default:
        default = None
    return default



with open("./defaults.json") as file:
    default = json.load(file)


def train_model():
    ...


# Parameter configuration
parser = argparse.ArgumentParser(
    # prog=scvae.__name__,
    # description=scvae.__description__,
    formatter_class=scvae.cli.argparse.ArgumentDefaultsHelpFormatter
)
subparsers = parser.add_subparsers(help="commands", dest="command")
subparsers.required = True
parser_train = subparsers.add_parser(
    name="train",
    description="Train model on single-cell transcript counts.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

data_set_subparsers = []
model_subparsers = []
training_subparsers = []
parser_train.set_defaults(func=train)
data_set_subparsers.append(parser_train)
model_subparsers.append(parser_train)
training_subparsers.append(parser_train)

for subparser in data_set_subparsers:
    subparser.add_argument(
        dest="data_set_file_or_name",
        help="data set name or path to data set file"
    )
    subparser.add_argument(
        "--format", "-f",
        dest="data_format",
        metavar="FORMAT",
        default=_parse_default(defaults["data"]["format"]),
        help="format of the data set"
    )
    subparser.add_argument(
        "--data-directory", "-D",
        metavar="DIRECTORY",
        default=_parse_default(defaults["data"]["directory"]),
        help="directory where data are placed or copied"
    )
    subparser.add_argument(
        "--map-features",
        action="store_true",
        default=_parse_default(defaults["data"]["map_features"]),
        help="map features using a feature mapping, if available"
    )
    subparser.add_argument(
        "--feature-selection", "-F",
        metavar="SELECTION",
        nargs="+",
        default=_parse_default(defaults["data"]["feature_selection"]),
        help="method for selecting features"
    )
    subparser.add_argument(
        "--example-filter", "-E",
        metavar="FILTER",
        nargs="+",
        default=_parse_default(defaults["data"]["example_filter"]),
        help=(
            "method for filtering examples, optionally followed by "
            "parameters"
        )
    )
    subparser.add_argument(
        "--preprocessing-methods", "-p",
        metavar="METHOD",
        nargs="+",
        default=_parse_default(defaults["data"]["preprocessing_methods"]),
        help="methods for preprocessing data (applied in order)"
    )
    subparser.add_argument(
        "--split-data-set",
        action="store_true",
        default=_parse_default(defaults["data"]["split_data_set"]),
        help="split data set into training, validation, and test sets"
    )
    subparser.add_argument(
        "--splitting-method",
        metavar="METHOD",
        default=_parse_default(defaults["data"]["splitting_method"]),
        help=(
            "method for splitting data into training, validation, and "
            "test sets"
        )
    )
    subparser.add_argument(
        "--splitting-fraction",
        metavar="FRACTION",
        type=float,
        default=_parse_default(defaults["data"]["splitting_fraction"]),
        help=(
            "fraction to use when splitting data into training, "
            "validation, and test sets"
        )
    )

for subparser in model_subparsers:
    subparser.add_argument(
        "--model-type", "-m",
        metavar="TYPE",
        default=_parse_default(defaults["models"]["type"]),
        help="type of model; either VAE or GMVAE"
    )
    subparser.add_argument(
        "--latent-size", "-l",
        metavar="SIZE",
        type=int,
        default=_parse_default(defaults["models"]["latent_size"]),
        help="size of latent space"
    )
    subparser.add_argument(
        "--hidden-sizes", "-H",
        metavar="SIZE",
        type=int,
        nargs="+",
        default=_parse_default(defaults["models"]["hidden_sizes"]),
        help="sizes of hidden layers"
    )
    subparser.add_argument(
        "--number-of-importance-samples",
        metavar="NUMBER",
        type=int,
        nargs="+",
        default=_parse_default(defaults["models"]["number_of_samples"]),
        help=(
            "the number of importance weighted samples (if two numbers "
            "are given, the first will be used for training and the "
            "second for evaluation)"
        )
    )
    subparser.add_argument(
        "--number-of-monte-carlo-samples",
        metavar="NUMBER",
        type=int,
        nargs="+",
        default=_parse_default(defaults["models"]["number_of_samples"]),
        help=(
            "the number of Monte Carlo samples (if two numbers are given, "
            "the first will be used for training and the second for "
            "evaluation)"
        )
    )
    subparser.add_argument(
        "--inference-architecture",
        metavar="KIND",
        default=_parse_default(defaults["models"][
                                   "inference_architecture"]),
        help="architecture of the inference model"
    )
    subparser.add_argument(
        "--latent-distribution", "-q",
        metavar="DISTRIBUTION",
        help=(
            "distribution for the latent variable(s); defaults depends on "
            "model type")
    )
    subparser.add_argument(
        "--number-of-classes", "-K",
        metavar="NUMBER",
        type=int,
        help="number of proposed clusters in data set"
    )
    subparser.add_argument(
        "--parameterise-latent-posterior",
        action="store_true",
        default=_parse_default(defaults["models"][
                                   "parameterise_latent_posterior"]),
        help="parameterise latent posterior parameters, if possible"
    )
    subparser.add_argument(
        "--generative-architecture",
        metavar="KIND",
        default=_parse_default(defaults["models"][
                                   "generative_architecture"]),
        help="architecture of the generative model"
    )
    subparser.add_argument(
        "--reconstruction-distribution", "-r",
        metavar="DISTRIBUTION",
        default=_parse_default(defaults["models"][
                                   "reconstruction_distribution"]),
        help="distribution for the reconstructions"
    )
    subparser.add_argument(
        "--number-of-reconstruction-classes", "-k",
        metavar="NUMBER",
        type=int,
        default=_parse_default(defaults["models"][
                                   "number_of_reconstruction_classes"]),
        help="the maximum count for which to use classification"
    )
    subparser.add_argument(
        "--prior-probabilities-method",
        metavar="METHOD",
        default=_parse_default(defaults["models"][
                                   "prior_probabilities_method"]),
        help="method to set prior probabilities"
    )

    subparser.add_argument(
        "--number-of-warm-up-epochs", "-w",
        metavar="NUMBER",
        type=int,
        default=_parse_default(defaults["models"][
                                   "number_of_warm_up_epochs"]),
        help=(
            "number of initial epochs with a linear weight on the "
            "KL divergence")
    )
    subparser.add_argument(
        "--kl-weight",
        metavar="WEIGHT",
        type=float,
        default=_parse_default(defaults["models"]["kl_weight"]),
        help="weighting of KL divergence"
    )
    subparser.add_argument(
        "--proportion-of-free-nats-for-y-kl-divergence",
        metavar="PROPORTION",
        type=float,
        default=_parse_default(defaults["models"][
                                   "proportion_of_free_nats_for_y_kl_divergence"]),
        help=(
            "proportion of maximum y KL divergence, which has constant "
            "term and zero gradients, for the GMVAE (free-bits method)"
        )
    )
    subparser.add_argument(
        "--minibatch-normalisation", "-b",
        action="store_true",
        default=_parse_default(defaults["models"][
                                   "minibatch_normalisation"]),
        help="use batch normalisation for minibatches in models"
    )
    subparser.add_argument(
        "--batch-correction", "--bc",
        action="store_true",
        default=_parse_default(defaults["models"][
                                   "batch_correction"]),
        help="use batch correction in models"
    )
    subparser.add_argument(
        "--dropout-keep-probabilities",
        metavar="PROBABILITY",
        type=float,
        nargs="+",
        default=_parse_default(defaults["models"][
                                   "dropout_keep_probabilities"]),
        help=(
            "list of probabilities, p, of keeping connections when using "
            "dropout (interval: ]0, 1[, where p in {0, 1, False} means no "
            "dropout)"
        )
    )
    subparser.add_argument(
        "--count-sum",
        action="store_true",
        default=_parse_default(defaults["models"]["count_sum"]),
        help="use count sum"
    )
    subparser.add_argument(
        "--minibatch-size", "-B",
        metavar="SIZE",
        type=int,
        default=_parse_default(defaults["models"]["minibatch_size"]),
        help="minibatch size for stochastic optimisation algorithm"
    )
    subparser.add_argument(
        "--run-id",
        metavar="ID",
        type=str,
        default=_parse_default(defaults["models"]["run_id"]),
        help=(
            "ID for separate run of the model (can only contain "
            "alphanumeric characters)"
        )
    )
    subparser.add_argument(
        "--models-directory", "-M",
        metavar="DIRECTORY",
        default=_parse_default(defaults["models"]["directory"]),
        help="directory where models are stored"
    )

for subparser in training_subparsers:
    subparser.add_argument(
        "--number-of-epochs", "-e",
        metavar="NUMBER",
        type=int,
        default=_parse_default(defaults["models"]["number_of_epochs"]),
        help="number of epochs for which to train"
    )
    subparser.add_argument(
        "--learning-rate",
        metavar="RATE",
        type=float,
        default=_parse_default(defaults["models"]["learning_rate"]),
        help="learning rate when training"
    )
    subparser.add_argument(
        "--new-run",
        action="store_true",
        default=_parse_default(defaults["models"]["new_run"]),
        help="train a model anew as a separate run with a generated run ID"
    )
    subparser.add_argument(
        "--reset-training",
        action="store_true",
        default=_parse_default(defaults["models"]["reset_training"]),
        help="reset already trained model"
    )
    subparser.add_argument(
        "--caches-directory", "-C",
        metavar="DIRECTORY",
        help="directory for temporary storage"
    )
    subparser.add_argument(
        "--analyses-directory", "-A",
        metavar="DIRECTORY",
        default=None,
        help="directory where analyses are saved"
    )

# Train the scVAE model
arguments = parser.parse_args()
model = arguments.func(**vars(arguments))
generated_data = model.sample(sample_size=1500)[0]
generated_data_values = generated_data.values

# # Save prediction
# with loompy.connect("./gene_by_cell_testing_all.loom") as data_file:
#     values = data_file[:, :].T

# ------------------------------------------------------
mean_gene_exp_pred = np.mean(generated_data_values, axis=0)
mean_gene_exp_labels = np.mean(values, axis=0)
test_pcc, _ = pearsonr(mean_gene_exp_pred, mean_gene_exp_labels)
test_scc, _ = spearmanr(mean_gene_exp_pred, mean_gene_exp_labels)
print('Test PCC:', test_pcc)
print('Test SCC:', test_scc)

test_pcc_str = str(np.round(test_pcc, 3))
test_scc_str = str(np.round(test_scc, 3))
cc_str = (
    f"Test PCC = {test_pcc_str}\n"
    f"Test SCC = {test_scc_str}"
)

scores_and_labels = np.concatenate((values, generated_data_values), axis=0)
# model = umap.UMAP().fit(values)
model = umap.UMAP().fit(scores_and_labels)
emedding = model.transform(scores_and_labels)
plt.scatter(emedding[:1500, 0], emedding[:1500, 1], color='blue', alpha=0.5, s=5, label='True')
plt.scatter(emedding[1500:, 0], emedding[1500:, 1], color='orange', alpha=0.5, s=5, label='Simulated')
plt.legend()
plt.title('UMAP', fontsize=15)
plt.xticks([])
plt.yticks([])
plt.xlabel('Dimension 1', fontsize=15)
plt.ylabel('Dimension 2', fontsize=15)
plt.show()
