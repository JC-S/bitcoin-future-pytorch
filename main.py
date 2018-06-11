import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from utils import load_data, use_cuda
from model import BiLSTM
from tqdm import tqdm
from parameters import *

def predict(model, test_dataloader):
    all_labels = []
    all_outputs = []

    pbar = tqdm(test_dataloader, total=len(test_dataloader))
    for inputs, labels in pbar:
        batch_len = len(inputs)
        model.hidden_0 = model.init_hidden_0(batch_len)
        model.hidden_1 = model.init_hidden_1(batch_len)
        model.hidden_2 = model.init_hidden_2(batch_len)
        all_labels.append(labels)
        if use_cuda:
            inputs = inputs.cuda()
        
        outputs = model(inputs)
        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_cuda:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs  

def train(model, train_dataloader, valid_dataloader, epochs=NB_EPOCH, lr=LR):
    criterion = nn.MSELoss()
    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    print('Model summary:')
    print(model)

    min_loss = float('inf')
    patience = 0

    for epoch in range(1, epochs+1):
        print(f'Running epoch: {epoch}')
        if patience == PATIENCE_MAX:
            patience = 0
            model = torch.load('best_model.pt')
            if use_cuda:
                model = model.cuda()
            lr /= 10
            print(f'Reset learning rate to {lr:.15f}')
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0001)

        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for inputs, labels in pbar:
            batch_len = len(inputs)
            model.hidden_0 = model.init_hidden_0(batch_len)
            model.hidden_1 = model.init_hidden_1(batch_len)
            model.hidden_2 = model.init_hidden_2(batch_len)
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_description(f'Loss = {loss.data:.5f}')
        print(f'Epoch {epoch}: {loss.data:.5f}')

        labels, outputs = predict(model, valid_dataloader)
        for i in range(len(outputs)):
            outputs[i] = 0 if outputs[i] < 0.5 else 1
        [labels, outputs] = [Variable(i) for i in [labels, outputs]]
        accuracy = torch.mean((outputs!=labels).float())
        print(f'Validation accuracy: {accuracy:.5f}')

        if loss.data < min_loss:
            min_loss = loss.data
            patience = 0
            print(f'Best model with loss {loss.data:.5f} found! Saving the model.')
            torch.save(model, 'best_model.pt')
        else:
            patience += 1
    
    return min_loss

def main():
    X_train, Y_train, X_valid, Y_valid, timestamp, close_prices = load_data('data.csv', TIME_WINDOW)
    [X_train, Y_train, X_valid, Y_valid] = [torch.from_numpy(i.astype(np.float32)) for i in [X_train, Y_train, X_valid, Y_valid]]
    model = BiLSTM(feature_num=FEATURE_NUM, time_window=TIME_WINDOW-1)
    dataset_train = torch.utils.data.TensorDataset(X_train, Y_train)
    dataset_valid = torch.utils.data.TensorDataset(X_valid, Y_valid)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
    valid_dataloader = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
    min_loss = train(model, train_dataloader, valid_dataloader)
    print(f'Best trained model has a loss of {min_loss:.5f}.')


if __name__ == '__main__':
    main()