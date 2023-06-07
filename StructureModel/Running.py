'''
Description:
    Codes for training and testing the VAE model.

Author:
    Jiaqi Zhang
'''
import numpy as np
import torch
from scipy.sparse import coo_matrix

import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("./StructureModel/")
from Py_Utils import prepareAugmentDataset, prepareDataset, getTimeStr, addDefaultArg, to_cpu_npy
from StructureVAE import VAE

# from Models.Py_Utils import prepareAugmentDataset, prepareDataset, getTimeStr, addDefaultArg, to_cpu_npy
# from Models.StructureModel.StructureVAE import VAE


def train_model(train_set, valid_set, max_epoch, device, num_layers, layer_size_list, cluster_weight_type, beta, learning_rate):
    '''
    Description:
        Train VAE model.
    '''
    # Construct the model and optimizer
    model = VAE(num_layers, layer_size_list, cluster_weight_type, beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # Train the model
    train_loss_list = []
    valid_loss_list = []
    # iterate through epochs
    for epoch in np.arange(max_epoch):
        print("-" * 70)
        print('Epoch ', epoch)
        train_loss = 0
        valid_loss = 0
        # -----
        model.train()
        train_pred_list = []
        train_labels_list = []
        # iterate through batches
        for x_batch, _ in train_set:
            x_batch = x_batch.to(device)
            # forward
            optimizer.zero_grad()
            forward_scores, z_mu, z_logvar = model(x_batch)
            # backward
            loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            # summary
            train_loss += loss.item()
            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            train_pred_list.append(to_cpu_npy(pred))
            train_labels_list.append(to_cpu_npy(x_batch))
        train_loss_list.append(train_loss / len(train_labels_list))
        # evaluate on validating data
        model.eval()
        valid_pred_list = []
        valid_labels_list = []
        with torch.no_grad():
            for x_batch, _ in valid_set:
                x_batch = x_batch.to(device)
                forward_scores, z_mu, z_logvar = model(x_batch)
                loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
                valid_loss += loss.item()
                pred = torch.reshape(forward_scores, (len(forward_scores), -1))
                valid_pred_list.append(to_cpu_npy(pred))
                valid_labels_list.append(to_cpu_npy(x_batch))
        valid_loss_list.append(valid_loss / len(valid_labels_list))
        print("Avg Training Loss = {:.3f} | Avg Validating Loss = {:.3f}".format(train_loss_list[-1], valid_loss_list[-1]))
    return model, train_loss_list, valid_loss_list


def eval_model(model, test_set, device):
    '''
    Description:
        Evaluate the model.
    '''
    model.eval()
    test_loss = []
    test_pred_list = []
    test_labels_list = []
    with torch.no_grad():
        for x_batch, _ in test_set:
            x_batch = x_batch.to(device)
            forward_scores, z_mu, z_logvar = model(x_batch)
            loss = model.loss(forward_scores, x_batch, z_mu, z_logvar)
            test_loss.append(loss.item())
            pred = torch.reshape(forward_scores, (len(forward_scores), -1))
            test_pred_list.append(to_cpu_npy(pred))
            test_labels_list.append(to_cpu_npy(x_batch))
    test_scores = np.concatenate(test_pred_list)
    test_labels = np.concatenate(test_labels_list)
    avg_test_loss = np.mean(test_loss)
    print("Test loss = {}".format(avg_test_loss))
    return avg_test_loss, test_scores, test_labels


def generateData(model, train_data, num_samples, device):
    '''
    Description:
        Data simulation.
    '''
    model.eval()
    generated_list = []
    for i in np.random.choice(np.arange(train_data.shape[0]), num_samples, replace=True):
        generated_list.append(model.generate(train_data[i].to(device))[0].cpu().detach().numpy())
    return np.asarray(generated_list)

# -------------------------------------------------------------------------

def clusterAugmentation(config):
    # Parameters configuration
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    beta = config["beta"]
    cluster_weight_type = config["cluster_weight_type"]
    layer_size_list = config["layer_size_list"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    need_save = config["need_save"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare datasets
    print("=" * 70)
    print("START LOADING DATA...")
    sampled_set, sampled_data, removed_data = prepareAugmentDataset(config["data"], batch_size, config["train_size"])
    # Train the model
    print("=" * 70)
    print("START TRAINING...")
    model, train_loss_list, valid_loss_list = train_model(
        sampled_set, sampled_set,
        max_epoch, device, num_layers, layer_size_list, cluster_weight_type, beta, learning_rate
    )
    torch.save(model.state_dict(), config["model_save_path"])
    summary = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
    }
    # Evaluate the model
    print("=" * 70)
    # print("START TESTING...")
    # test_loss, test_scores, test_labels = eval_model(model, sampled_set, device)
    # Data simulation with trained model
    print("=" * 70)
    print("START GENERATING...")
    generated_data = generateData(model, torch.Tensor(sampled_data), removed_data.shape[0], device)
    print("Generated data shape : {}".format(generated_data))
    # Save records
    # summary["test_loss"] = test_loss
    # summary["test_predictions"] = test_scores
    # summary["test_labels"] = test_labels
    summary["generated_data"] = generated_data
    summary["sampled_data"] = sampled_data
    summary["removed_data"] = removed_data
    if need_save:
        np.save(config["prediction_save_path"], summary)
        print("Finished saving records.")
    return summary


def normalAugmentation(config):
    # Parameters configuration
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    beta = config["beta"]
    cluster_weight_type = config["cluster_weight_type"]
    layer_size_list = config["layer_size_list"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    need_save = config["need_save"]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Prepare datasets
    print("=" * 70)
    print("START LOADING DATA...")
    train_set, valid_set, test_set  = prepareDataset(
        config["train_data"], config["validate_data"], config["test_data"],
        batch_size
    )
    # Train the model
    print("=" * 70)
    print("START TRAINING...")
    model, train_loss_list, valid_loss_list = train_model(
        train_set, valid_set,
        max_epoch, device, num_layers, layer_size_list, cluster_weight_type, beta, learning_rate
    )
    summary = {
        "train_loss_list" : train_loss_list,
        "valid_loss_list" : valid_loss_list,
    }
    # Test the model
    print("=" * 70)
    print("START TESTING...")
    test_loss, test_scores, test_labels = eval_model(model, test_set, device)
    summary["test_loss"] = test_loss
    summary["test_predictions"] = coo_matrix(test_scores)
    summary["test_labels"] = coo_matrix(test_labels)
    if need_save:
        torch.save(model.state_dict(), config["model_save_path"])
        np.save(config["prediction_save_path"], summary)
        print("Finished saving records.")
    return summary

# -------------------------------------------------------------------------

def modelTrainForCV(config):
    # Parameters configuration
    max_epoch = config["max_epoch"]
    beta = config["beta"]
    cluster_weight_type = config["cluster_weight_type"]
    layer_size_list = config["layer_size_list"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Prepare datasets
    train_set, valid_set, test_set = config["train_data"], config["validate_data"], config["test_data"]
    # Train the model
    print("=" * 70)
    print("START TRAINING...")
    model, _, _ = train_model(
        train_set, valid_set,
        max_epoch, device, num_layers, layer_size_list, cluster_weight_type, beta, learning_rate
    )
    # Test the model
    print("=" * 70)
    print("START TESTING ON VALIDATE DATA ...")
    _, validate_scores, validate_labels = eval_model(model, valid_set, device)
    return validate_scores, validate_labels