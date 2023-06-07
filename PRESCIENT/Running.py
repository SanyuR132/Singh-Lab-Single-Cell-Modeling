# training on veres et al. dataset
import copy
import pickle as pkl

import prescient.train as train
import prescient.simulate as traj
from prescient.train.model import *

# --------------------------------------------------

def init_config(args):

    config = SimpleNamespace(

        seed = args.seed,
        timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime()),

        # data parameters
        data_path = args.data_path,
        weight = args.weight,

        # model parameters
        activation = args.activation,
        layers = args.layers,
        k_dim = args.k_dim,

        # pretraining parameters
        pretrain_burnin = 50,
        pretrain_sd = 0.1,
        pretrain_lr = 1e-9,
        pretrain_epochs = args.pretrain_epochs,

        # training parameters
        train_dt = args.train_dt,
        train_sd = args.train_sd,
        train_batch_size = args.train_batch,
        ns = 2000,
        train_burnin = 100,
        train_tau = args.train_tau,
        train_epochs = args.train_epochs,
        train_lr = args.train_lr,
        train_clip = args.train_clip,
        save = args.save,

        # loss parameters
        sinkhorn_scaling = 0.7,
        sinkhorn_blur = 0.1,

        # file parameters
        out_dir = args.out_dir,
        out_name = args.out_dir.split('/')[-1],
        pretrain_pt = os.path.join(args.out_dir, 'pretrain.pt'),
        train_pt = os.path.join(args.out_dir, 'train.{}.pt'),
        train_log = os.path.join(args.out_dir, 'train.log'),
        done_log = os.path.join(args.out_dir, 'done.log'),
        config_pt = os.path.join(args.out_dir, 'config.pt'),
    )

    config.train_t = []
    config.test_t = []

    if not os.path.exists(args.out_dir):
        print('Making directory at {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    else:
        print('Directory exists at {}'.format(args.out_dir))
    return config


def load_data(args):
    return torch.load(args.data_path)


def train_init(args):

    a = copy.copy(args)

    # data
    data_pt = load_data(args)
    if np.all(np.isnan(data_pt["w"][0])):
        for i in range(len(data_pt["w"])):
            # data_pt["w"][i] = np.ones((len(data_pt["w"][i]), )) * 0.1
            data_pt["w"][i] = np.ones((len(data_pt["w"][i]), ))
    x = data_pt["xp"]
    y = data_pt["y"]
    weight = data_pt["w"]
    if args.weight_name != None:
        a.weight = args.weight_name
    # weight = os.path.basename(a.weight_path)
    # weight = weight.split('.')[0].split('-')[-1]


    # out directory
    a.train_sd = args.train_sd
    a.train_lr = args.train_lr
    a.train_clip = args.train_clip
    a.train_batch = args.train_batch
    a.tune_metric = args.tune_metric
    a.tune_day = args.tune_day
    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}-"
        "{train_sd}-"
        "{train_lr}-"
        "{train_clip}-"
        "{train_batch}-"
        "{tune_metric}-"
        "day{tune_day}"
    ).format(**a.__dict__)
    args.__delattr__("tune_metric")
    args.__delattr__("tune_day")

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed))
    config = init_config(a)

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]

    config.start_t = y[0]
    config.train_t = y[1:]
    y_start = y[config.start_t]
    y_ = [y_ for y_ in y if y_ > y_start]

    w_ = weight[config.start_t]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start)*w_)) for yy in y_}

    return x, y, w, config


def trainModel(args):
    train.run(args, train_init)

# --------------------------------------------------

def makeSimulation(args):
    # load data
    data_pt = torch.load(args.data_path)
    expr = data_pt["data"]
    pca = data_pt["pca"]
    xp = pca.transform(expr)
    # xp = expr

    # torch device
    if args.gpu != None:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # load model
    config_path = os.path.join(str(args.model_path), 'seed_{}/config.pt'.format(args.seed))
    config = SimpleNamespace(**torch.load(config_path))
    net = AutoGenerator(config)

    # train_pt = os.path.join(args.model_path, 'seed_{}/train.epoch_{}.pt'.format(args.seed, args.epoch))
    train_pt = os.path.join(args.model_path, 'seed_{}/train.best.pt'.format(args.seed, args.epoch))
    checkpoint = torch.load(train_pt, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.to(device)

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))
    else:
        num_steps = int(args.num_steps)

    # simulate forward
    num_cells = min(args.num_cells, xp.shape[0])
    out = traj.simulate(xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims, num_cells, num_steps, device, args.tp_subset, args.celltype_subset)

    # write simulation data to file
    out_path = os.path.join(args.out_path, args.model_path.split("/")[-1], 'seed_{}_train.sims_{}_num.cells_{}_num.steps_{}_subsets_{}_{}_simulation.pt'.format(args.seed, args.num_sims, args.num_cells, num_steps, args.tp_subset, args.celltype_subset))
    torch.save({
    "sims": out
    }, out_path)

    out_path = os.path.join(args.out_path, args.model_path.split("/")[-1],
                            'seed_{}_train.sims_{}_num.cells_{}_num.steps_{}_subsets_{}_{}_simulation.pkl'.format(
                                args.seed, args.num_sims, args.num_cells, num_steps, args.tp_subset,
                                args.celltype_subset))
    with open(out_path, "wb") as file:
        pkl.dump(out, file)

# --------------------------------------------------

def modelTrainForCV(train_ags, simulate_args, metric=None, day=None):
    # Train the model
    args = train_ags

    args.tune_metric = metric
    args.tune_day = day

    train.run(args, train_init)
    # -----
    # Make simulations
    args = simulate_args
    # load data
    data_pt = torch.load(args.data_path)
    expr = data_pt["data"]
    pca = data_pt["pca"]
    xp = pca.transform(expr)
    # xp = expr
    # torch device
    if args.gpu != None:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # load model
    config_path = os.path.join(str(args.model_path), 'seed_{}/config.pt'.format(args.seed))
    config = SimpleNamespace(**torch.load(config_path))
    net = AutoGenerator(config)

    # train_pt = os.path.join(args.model_path, 'seed_{}/train.epoch_{}.pt'.format(args.seed, args.epoch))
    train_pt = os.path.join(args.model_path, 'seed_{}/train.best.pt'.format(args.seed, args.epoch))
    checkpoint = torch.load(train_pt, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.to(device)

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))
    else:
        num_steps = int(args.num_steps)

    # simulate forward
    # num_cells = min(args.num_cells, xp.shape[0])
    num_cells = xp.shape[0]
    # out = traj.simulate(xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims,
    #                     args.num_cells, num_steps, device, args.tp_subset, args.celltype_subset)
    out = traj.simulate(xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims,
                        num_cells, num_steps, device, args.tp_subset, args.celltype_subset)
    return out[0][-1, :, :], xp
