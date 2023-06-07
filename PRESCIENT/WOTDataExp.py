import argparse
from scipy.io import mmread
from Running import trainModel, makeSimulation
import os

# =====================================================

def create_train_parser(pred_day):
    if "before{}".format(pred_day) not in os.listdir('../../Prediction/PRESCIENT/'):
        os.mkdir('../../Prediction/PRESCIENT/before{}'.format(pred_day))

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true')
    parser.add_argument('--gpu', default = 7, type = int, help="Designate GPU number as an integer (compatible with CUDA).")
    parser.add_argument('--out_dir', default = '../../Prediction/PRESCIENT/before{}'.format(pred_day), help="Directory for storing training output.")
    parser.add_argument('--seed', type = int, default = 2, help="Set seed for training process.")
    # -- data options
    parser.add_argument('-i', '--data_path', default = "../../Data/WOT/pt_data/before_day{}_data.pt".format(pred_day), help="Input PRESCIENT data torch file.")
    parser.add_argument('--weight_name', default = "kegg-growth", help="Designate descriptive name of growth parameters for filename.")
    # -- model options
    parser.add_argument('--loss', default = 'euclidean', help="Designate distance function for loss.")
    parser.add_argument('--k_dim', default = 500, type = int, help="Designate hidden units of NN.")
    parser.add_argument('--activation', default = 'softplus', help="Designate activation function for layers of NN.")
    parser.add_argument('--layers', default = 1, type = int, help="Choose number of layers for neural network parameterizing the potential function.")
    # -- pretrain options
    parser.add_argument('--pretrain_epochs', default = 10, type = int, help="Number of epochs for pretraining with contrastive divergence.")
    # -- train options
    parser.add_argument('--train_epochs', default = 200, type = int, help="Number of epochs for training.")
    parser.add_argument('--train_lr', default = 1e-4, type = float, help="Learning rate for Adam optimizer during training.")
    parser.add_argument('--train_dt', default = 0.1, type = float, help="Timestep for simulations during training.")
    parser.add_argument('--train_sd', default = 0.5, type = float, help="Standard deviation of Gaussian noise for simulation steps.")
    parser.add_argument('--train_tau', default = 1e-6, type = float, help="Tau hyperparameter of PRESCIENT.")
    parser.add_argument('--train_batch', default = 0.1, type = float, help="Batch size for training.")
    parser.add_argument('--train_clip', default = 0.25, type = float, help="Gradient clipping threshold for training.")
    parser.add_argument('--save', default = 100, type = int, help="Save model every n epochs.")
    # -- run options
    parser.add_argument('--pretrain', type=bool, default=True, help="If True, pretraining will run.")
    parser.add_argument('--train', type=bool, default=True, help="If True, training will run with existing pretraining torch file.")
    parser.add_argument('--config')
    return parser


def create_simulate_parser(pred_day):
    data = mmread("../../Data/WOT/pt_data/day{}_data.mtx".format(pred_day))
    cur_size = data.shape
    print("Simulation size = {}".format(cur_size))

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", default="../../Data/WOT/pt_data/before_day{}_data.pt".format(pred_day), required=False, help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path", default="../../Prediction/PRESCIENT/before{}/kegg-growth-softplus_1_500-1e-06/".format(pred_day), required=False, help="Path to directory containing PRESCIENT model for simulation.")
    parser.add_argument("--seed", default=2, required=False, help="Choose the seed of the trained model to use for simulations.")
    parser.add_argument("--epoch", type=str, required=False, help="Choose which epoch of the model to use for simulations.")
    parser.add_argument("--num_sims", default=1, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=cur_size[0], help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=1, required=False, help="Define number of forward steps of size dt to take.")
    parser.add_argument("--gpu", default=None, required=False, help="If available, assign GPU device number.")
    parser.add_argument("--celltype_subset", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata.")
    parser.add_argument("--tp_subset", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")
    parser.add_argument("-o", "--out_path", required=False, default="../../Prediction/PRESCIENT/before{}".format(pred_day), help="Path to output directory.")
    return parser


# =====================================================
day_list = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
for each in day_list:
    train_parser = create_train_parser(str(each))
    args = train_parser.parse_args()
    trainModel(args)
    # -----
    simulation_parser = create_simulate_parser(str(each))
    args = simulation_parser.parse_args()
    makeSimulation(args)
