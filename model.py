import torch
import torch.nn as nn
from torch.autograd import Variable
from parameters import *
from utils import use_cuda

feature_mul = 2 if BIODIRECT else 1

class BiLSTM(nn.Module):
    def __init__(self, feature_num=1, time_window=10):
        super(BiLSTM, self).__init__()
        self.feature_num = feature_num
        self.time_window = time_window
        self.lstm0 = nn.LSTM(input_size=self.feature_num,
                             hidden_size=self.feature_num,
                             batch_first=True,
                             bidirectional=BIODIRECT)
        self.lstm1 = nn.LSTM(input_size=self.feature_num*feature_mul,
                             hidden_size=self.feature_num*2,
                             batch_first=True,
                             bidirectional=BIODIRECT)
        self.lstm2 = nn.LSTM(input_size=self.feature_num*2*feature_mul,
                             hidden_size=1,
                             batch_first=True,
                             bidirectional=BIODIRECT)
        self.drop = nn.Dropout(0.6)
        #self.linear = nn.Linear(self.time_window*feature_mul, 1)
        self.linear = nn.Linear(self.time_window*feature_mul*self.feature_num, 1)

    def init_hidden_0(self, batch_len):
        if use_cuda:
            hidden_0 = (torch.zeros(1, batch_len, self.feature_num).cuda(),
                        torch.zeros(1, batch_len, self.feature_num).cuda())
        else:
            hidden_0 = (torch.zeros(1, batch_len, self.feature_num),
                        torch.zeros(1, batch_len, self.feature_num))
        return hidden_0

    def init_hidden_1(self, batch_len):
        if use_cuda:
            hidden_1 = (torch.zeros(1, batch_len, self.feature_num*2).cuda(),
                        torch.zeros(1, batch_len, self.feature_num*2).cuda())
        else:
            hidden_1 = (torch.zeros(1, batch_len, self.feature_num*2),
                        torch.zeros(1, batch_len, self.feature_num*2))
        return hidden_1

    def init_hidden_2(self, batch_len):
        if use_cuda:
            hidden_2 = (torch.zeros(1, batch_len, 1).cuda(),
                        torch.zeros(1, batch_len, 1).cuda())
        else:
            hidden_2 = (torch.zeros(1, batch_len, 1),
                        torch.zeros(1, batch_len, 1))
        return hidden_2
        
    def forward(self, x):
        x, self.hidden_0 = self.lstm0(x, self.hidden_0)
        x = self.drop(x)
        #x, self.hidden_1 = self.lstm1(x, self.hidden_1)
        #x = self.drop(x)
        #x, self.hidden_2 = self.lstm2(x, self.hidden_2)
        #x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x.view(-1)