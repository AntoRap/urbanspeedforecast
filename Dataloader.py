import numpy as np
from itertools import chain
import glob

import pandas as pd
import pickle
import torch


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_data(data):
    if data == "PEMS03":
        x = np.load("data/PEMS03.npz")["data"][:, :, 0:1]
        adj_mx = np.load("data/adj.npy")
        # reduced_adjmatrix = adj_mx[0:10, 0:10]
        # print(reduced_adjmatrix.shape)
        # subset = x[:, 0:10, :]
        print(x.shape)
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "PEMS08":
        x = np.load("data/PEMS08.npz")["data"][:, :, 0:1]
        adj_mx = np.load("data/adj_PEMS08.npy")
        # reduced_adjmatrix = adj_mx[0:10, 0:10]
        # print(reduced_adjmatrix.shape)
        # subset = x[:, 0:10, :]
        print(x.shape)
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "Lyon2019_J_to_M_247":
        x = np.load("data//Lyon2019_J_to_M_247.npz")["data"][:, :]
        H = 1
        B, N = x.shape
        x = x.reshape(B, N)
        adj_mx = np.load("data//Lyon_AdjMatrix_247.npy")
    elif data == "Lyon2019_J_to_M_183":
        x = np.load("data//Lyon2019_J_to_M_183.npz")["data"][:, :]
        H = 1
        B, N = x.shape
        x = x.reshape(B, N)
        adj_mx = np.load("data//Lyon_AdjMatrix_183.npy")
    elif data == "Lyon2019_J_to_M":
        adj_mx = np.load("data//Lyon_AdjMatrix.npy")
        adj_mx = adj_mx[0:10, 0:10]
        x = np.load("data//Lyon2019_J_to_M.npz")["data"][:, 0:10]
        H = 1
        B, N = x.shape
        x = x.reshape(B, N)
    elif data == "flow":
        x = np.load("data//flow_j.npz")["data"][:, :, 0:800]
        H = 1
        B, H, N = x.shape
        x = x.reshape(B, N)

        adj_mx = np.load("data//adj.npy")
        adj_mx = adj_mx[0:800, 0:800]

    adj = torch.tensor(np.array(adj_mx), dtype=torch.float32)
    return x, adj


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    # print(time_len,rate,data.shape)

    train_size = int(time_len * rate)
    train_data = data[0:train_size]  # seqlen * num nodi
    val_data = data[train_size:int(time_len * (rate + 0.2))]
    test_data = data[int(time_len * (rate + 0.2)):time_len]

    trainX, trainY, valX, valY, testX, testY = [], [], [], [], [], []
    i = 0
    # print(len(train_data) - seq_len - pre_len)
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        # print(a[12:24,0])#[ 90.][109.][ 90.][ 90.] [111.][107.666664][109.][ 88.][ 81.][ 98.][ 90.][ 90. ]]
        # print("stop")
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(val_data) - seq_len - pre_len):
        c = val_data[i: i + seq_len + pre_len]
        valX.append(c[0: seq_len])
        valY.append(c[seq_len: seq_len + pre_len])

    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)

    trainY1 = np.array(trainY)

    testX1 = np.array(testX)
    testY1 = np.array(testY)

    #train (3000, 12, 247)
    #val (984, 12, 247)
    #test (984, 12, 247)

    valX1 = np.array(valX)
    valY1 = np.array(valY)

    mean, std = np.mean(trainX1), np.std(trainX1)
    trainX1 = (trainX1 - mean) / std
    valX1 = (valX1 - mean) / std
    testX1 = (testX1 - mean) / std

    trainX1 = torch.tensor(trainX1, dtype=torch.float32)
    trainY1 = torch.tensor(trainY1, dtype=torch.float32)
    testX1 = torch.tensor(testX1, dtype=torch.float32)
    testY1 = torch.tensor(testY1, dtype=torch.float32)
    valX1 = torch.tensor(valX1, dtype=torch.float32)
    valY1 = torch.tensor(valY1, dtype=torch.float32)

    return trainX1, trainY1, valX1, valY1, testX1, testY1, mean, std