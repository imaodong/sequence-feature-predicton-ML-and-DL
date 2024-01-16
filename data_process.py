import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import MinMaxScaler


def generate_data(mod,dataset):
   
    datas,labels = [],[]
    for i in range(0,len(dataset)+1-mod*2):
        datas.append(dataset[i:i+mod])
        labels.append(dataset[i+mod:i+mod+mod])
    datas,labels = np.array(datas,dtype=float),np.array(labels,dtype=float)

    return torch.tensor(datas).float(),torch.tensor(labels).float()


def read_data():
    # 0: time, 1~6: feature, 7: label
    train_d = pd.read_csv('./train_set.csv').values # 8640
    valid_d = pd.read_csv('./validation_set.csv').values # 2976
    test_d = pd.read_csv('./test_set.csv').values # 2976
    scaler = MinMaxScaler()

    train_date,valid_date,test_date = train_d[:,0], valid_d[:,0],test_d[:,0]
    data_train,data_valid,data_test = scaler.fit_transform(train_d[:,1:]),scaler.fit_transform(valid_d[:,1:]),scaler.fit_transform(test_d[:,1:])

    return data_train,data_valid,data_test,train_date,valid_date,test_date


def get_loader(train_data,train_label,valid_data,valid_label,test_data,test_label,batch_size):
    train_loader = DataLoader(TensorDataset(train_data,train_label),batch_size,False)
    valid_loader = DataLoader(TensorDataset(valid_data,valid_label),batch_size,False)
    test_loader = DataLoader(TensorDataset(test_data,test_label),batch_size,False)

    return train_loader,valid_loader,test_loader




