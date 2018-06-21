import torch
import pandas as pd
import numpy as np

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

def load_data(target, sequence_length, start=0, shuffle=True):
    #Read the data file
    raw_data = pd.read_csv(target, nrows = 20000 ,dtype = float).values
    #Reverse the data so it'll be from early to late
    raw_data = np.flip(raw_data, axis=0)

    #Generate data sequence with time window = sequence_length
    raw_data = raw_data.tolist()
    data = []
    for i in range(len(raw_data)-sequence_length):
        data.append(raw_data[i:i+sequence_length])
    data = np.array(data)
    
    #Extract timestamp before normalization
    timestamp = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        timestamp[i] = data[i][0][0]
    
    #Normalize data to change percentage
    data_norm = np.zeros_like(data)
    for i0 in range(data.shape[0]):
        for i1 in range(data.shape[1]):
            data_norm[i0,i1] = data[i0,i1]/data[i0,i1-1]-1

    #The unnormalized close prices for Y_test
    close_prices = data[start:,0:1,-1]
    
    #Split dataset to training and validation
    split_line = round(0.9 * data_norm.shape[0])
    trainset = data_norm[:split_line]
    validset = data_norm[split_line:]
    
    #Final training data, last time frame is the target, remove the timestamp
    X_train = trainset[:,:-1,1:]
    Y_train = trainset[:,-1,4]
    #Normalize target to 1=positive 0=negative
    for i in range(Y_train.shape[0]):
        Y_train[i] = 1 if Y_train[i] > 0 else 0

    #Final testing data
    X_valid = validset[:,:-1,1:]
    Y_valid = validset[:,-1,4]

    #Normalize target to 1=positive 0=negative
    def _normalize_target(Array_in):
        for i in range(Array_in.shape[0]):
            Array_in[i] = 1 if Array_in[i] > 0 else 0
        return Array_in
    Y_train = _normalize_target(Y_train)
    Y_valid = _normalize_target(Y_valid)

    return X_train, Y_train, X_valid, Y_valid, timestamp, close_prices
