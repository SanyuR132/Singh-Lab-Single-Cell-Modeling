from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

from scipy.io import mmread
import numpy as np
import pandas as pd


def _loadData(train_filename, test_filename):
    train_data = np.asarray(mmread(train_filename).todense()).T
    test_data = np.asarray(mmread(test_filename).todense()).T
    return train_data, test_data


def _loadCellLabel(annotate_filename, train_cell_id_filename, test_cell_id_filename):
    annotate_data = pd.read_csv(annotate_filename, index_col=0, header=0)
    train_cell_id_data = pd.read_csv(train_cell_id_filename, skiprows=0)
    test_cell_id_data = pd.read_csv(test_cell_id_filename, skiprows=0)
    # -----
    train_annotate_data = annotate_data.loc[train_cell_id_data.values.squeeze()]
    train_cell_cluster = train_annotate_data.Main_Cluster.values
    train_cell_type = train_annotate_data.Main_cell_type.values
    # -----
    test_annotate_data = annotate_data.loc[test_cell_id_data.values.squeeze()]
    test_cell_cluster = test_annotate_data.Main_Cluster.values
    test_cell_type = test_annotate_data.Main_cell_type.values
    return train_cell_type, test_cell_type

# ==================================================================

def predict(train_data, train_label, test_data, model_type = "KNN"):
    # Create model
    if model_type == "KNN":
        model = KNeighborsClassifier()
    elif model_type == "LR":
        model = LogisticRegression()
    elif model_type == "SVM":
        model = SVC()
    elif model_type == "RF":
        model = RandomForestClassifier()
    elif model_type == "random":
        model = DummyClassifier()
    else:
        raise NotImplementedError("Not implemented model type : {}!".format(model_type))
    # -----
    # Train model
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    return pred_label


def evaluation(true_label, pred_label):
    f1 = f1_score(true_label, pred_label, average="weighted")
    return f1




if __name__ == '__main__':
    train_filename = "../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_9_5.mtx"
    train_cell_id_filename = "../Data/mouse_cell/wo_preprocess/cells_cnt_time_9_5.csv"

    test_filename = "../Data/mouse_cell/wo_preprocess/gene_cnt_mat_time_10_5.mtx"
    test_cell_id_filename = "../Data/mouse_cell/wo_preprocess/cells_cnt_time_10_5.csv"

    annotate_filename = "../Data/mouse_cell/cell_annotate.csv"

    model_type = "LR" # KNN, LR, SVM, RF
    # -----
    train_data, test_data = _loadData(train_filename, test_filename)
    print("Train data shape : {}".format(train_data.shape))
    print("Test data shape : {}".format(test_data.shape))
    # -----
    train_label, test_label = _loadCellLabel(annotate_filename, train_cell_id_filename, test_cell_id_filename)
    # -----
    # On train data
    model_pred_label = predict(train_data, train_label, test_data=train_data, model_type=model_type)
    random_pred_label = predict(train_data, train_label, test_data=train_data, model_type="random")
    model_f1 = evaluation(train_label, model_pred_label)
    random_f1 = evaluation(train_label, random_pred_label)
    print("-" * 70)
    print("[ Train Data ]")
    print("{} F1 = {} | Random F1 = {}".format(model_type, model_f1, random_f1))
    # -----
    # On test data
    model_pred_label = predict(train_data, train_label, test_data=test_data, model_type=model_type)
    random_pred_label = predict(train_data, train_label, test_data=test_data, model_type="random")
    model_f1 = evaluation(test_label, model_pred_label)
    random_f1 = evaluation(test_label, random_pred_label)
    print("-" * 70)
    print("[ Test Data ]")
    print("{} F1 = {} | Random F1 = {}".format(model_type, model_f1, random_f1))
