import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import Data


def load_data(args):
    adj, features, labels, idx_train, idx_valid, idx_test, sens, sens_idx = load(args.dataset)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    data = Data(x=features.to(args.device), edge_index=edge_index.to(args.device), input_dim=features.shape[1],
                idx_train_list=idx_train, idx_valid_list=idx_valid, idx_test_list=idx_test, y=labels.to(args.device),
                sens=sens.to(args.device), adj=adj)
    return data


def load(dataset_name):
    if dataset_name == 'credit':
        sens_attr = "Age" 
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = "data/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset_name, sens_attr, predict_attr,
                                                                                path=path_credit,
                                                                                label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset_name.split('_')[0] == 'pokec':
        if dataset_name == 'pokec_z':
            dataset_name = 'region_job'
        elif dataset_name == 'pokec_n':
            dataset_name = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 6000
        sens_idx = 3
        path_pokec = "data/pokec"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset_name, sens_attr, predict_attr,
                                                                               path=path_pokec,
                                                                               label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset_name == 'income':
        sens_attr = "race" 
        sens_idx = 8
        predict_attr = 'income'
        label_number = 1000
        path_income = "data/income"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_income(dataset_name, sens_attr, predict_attr,
                                                                                path=path_income,
                                                                                label_number=label_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    elif dataset_name in ['trenciansky', "presovsky", 'banskobystricky', 'kosicky']:
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 6000
        sens_idx = 3
        path = 'data/transfer/sample_data/'
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_transfer(dataset_name, sens_attr, predict_attr,
                                                                               path=path,
                                                                               label_number=label_number)  
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features    

    num_class = len(labels.unique())
    sens_1 = sum(sens == 1).item()
    sens_0 = sum(sens == 0).item()
    train_sens_1 = sum(sens[idx_train] == 1).item()
    train_sens_0 = sum(sens[idx_train] == 0).item()
    val_sens_1 = sum(sens[idx_val] == 1).item()
    val_sens_0 = sum(sens[idx_val] == 0).item()
    test_sens_1 = sum(sens[idx_test] == 1).item()
    test_sens_0 = sum(sens[idx_test] == 0).item()

    label_1 = sum(labels == 1).item()
    label_0 = sum(labels == 0).item()
    train_label_1 = sum(labels[idx_train] == 1).item()
    train_label_0 = sum(labels[idx_train] == 0).item()
    val_label_1 = sum(labels[idx_val] == 1).item()
    val_label_0 = sum(labels[idx_val] == 0).item()
    test_label_1 = sum(labels[idx_test] == 1).item()
    test_label_0 = sum(labels[idx_test] == 0).item()

    return adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def load_transfer(dataset, sens_attr, predict_attr, path, label_number=6000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}_processed.csv".format(dataset)))
    idx_features_labels[predict_attr] = idx_features_labels[predict_attr].fillna(-1)
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values.astype(int)
    labels[labels > 1] = 1

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/",
                label_number=3000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(dataset, sens_attr, predict_attr, path="../dataset/pokec/", label_number=6000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels > 1] = 1

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_income(dataset, sens_attr="race", predict_attr="income", path="../data/income/", label_number=1000):  # 1000
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map