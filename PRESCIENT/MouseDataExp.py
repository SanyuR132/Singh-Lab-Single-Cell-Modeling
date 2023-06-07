import argparse
from Running import trainModel, makeSimulation

# =====================================================

def create_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true')
    parser.add_argument('--gpu', default = 7, type = int, help="Designate GPU number as an integer (compatible with CUDA).")
    parser.add_argument('--out_dir', default = '../../Prediction/PRESCIENT/', help="Directory for storing training output.")
    parser.add_argument('--seed', type = int, default = 2, help="Set seed for training process.")
    # -- data options
    # parser.add_argument('-i', '--data_path', default = "./Veres2019_data.pt", help="Input PRESCIENT data torch file.")
    # parser.add_argument('-i', '--data_path', default = "../../Data/mouse_cell/pt_data/mouse_cell_data.pt", help="Input PRESCIENT data torch file.")
    parser.add_argument('-i', '--data_path', default = "../../Data/mouse_cell/pt_data/before_day11_data.pt", help="Input PRESCIENT data torch file.")
    parser.add_argument('--weight_name', default = "kegg-growth", help="Designate descriptive name of growth parameters for filename.")
    # -- model options
    parser.add_argument('--loss', default = 'euclidean', help="Designate distance function for loss.")
    parser.add_argument('--k_dim', default = 100, type = int, help="Designate hidden units of NN.")
    parser.add_argument('--activation', default = 'softplus', help="Designate activation function for layers of NN.")
    parser.add_argument('--layers', default = 2, type = int, help="Choose number of layers for neural network parameterizing the potential function.")
    # -- pretrain options
    parser.add_argument('--pretrain_epochs', default = 5, type = int, help="Number of epochs for pretraining with contrastive divergence.")
    # -- train options
    parser.add_argument('--train_epochs', default = 100, type = int, help="Number of epochs for training.")
    parser.add_argument('--train_lr', default = 0.1, type = float, help="Learning rate for Adam optimizer during training.")
    parser.add_argument('--train_dt', default = 0.1, type = float, help="Timestep for simulations during training.")
    parser.add_argument('--train_sd', default = 0.7000000000000001, type = float, help="Standard deviation of Gaussian noise for simulation steps.")
    parser.add_argument('--train_tau', default = 11405463308018093, type = float, help="Tau hyperparameter of PRESCIENT.")
    parser.add_argument('--train_batch', default = 0.1, type = float, help="Batch size for training.")
    parser.add_argument('--train_clip', default = 0.25, type = float, help="Gradient clipping threshold for training.")
    parser.add_argument('--save', default = 100, type = int, help="Save model every n epochs.")
    # -- run options
    parser.add_argument('--pretrain', type=bool, default=True, help="If True, pretraining will run.")
    parser.add_argument('--train', type=bool, default=True, help="If True, training will run with existing pretraining torch file.")
    parser.add_argument('--config')


    parser.add_argument('--tune_metric', type=str, default="KS")
    parser.add_argument('--tune_day', type=str, default="11")
    return parser


def create_simulate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", default="./mouse_cell_data.pt", required=False, help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path", default="../../Prediction/PRESCIENT/kegg-growth-softplus_1_500-1e-06/", required=False, help="Path to directory containing PRESCIENT model for simulation.")
    parser.add_argument("--seed", default=2, required=False, help="Choose the seed of the trained model to use for simulations.")
    parser.add_argument("--epoch", type=str, required=False, help="Choose which epoch of the model to use for simulations.")
    parser.add_argument("--num_sims", default=1, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=10000, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=1, required=False, help="Define number of forward steps of size dt to take.")
    parser.add_argument("--gpu", default=None, required=False, help="If available, assign GPU device number.")
    parser.add_argument("--celltype_subset", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata.")
    parser.add_argument("--tp_subset", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")
    parser.add_argument("-o", "--out_path", required=False, default="../../Prediction/PRESCIENT/", help="Path to output directory.")
    return parser


# =====================================================

train_parser = create_train_parser()
args = train_parser.parse_args()
trainModel(args)
# # -----
# simulation_parser = create_simulate_parser()
# args = simulation_parser.parse_args()
# makeSimulation(args)
