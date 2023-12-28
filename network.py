'''
Implementation of article https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf
'''
import torch
from torch import nn
from torch.nn import LSTM, Linear
from torch.utils.data import TensorDataset
import math

class FxDataset(TensorDataset):
    def __init__(self, x, y, window_len, batch_size):
        super(TensorDataset, self).__init__()

        len_index = len(x) - (len(x) % (window_len * batch_size))
        self.x = x[:len_index].reshape(-1, 1)
        self.y = y[:len_index].reshape(-1, 1)
        self.frame_size = window_len * batch_size
        self.window_len = window_len
        self.batch_size = batch_size

    def __getitem__(self, index):
        x_out = self.x[index*self.frame_size : (index+1)*self.frame_size].reshape(self.batch_size, self.window_len, 1)
        y_out = self.y[index*self.frame_size : (index+1)*self.frame_size].reshape(self.batch_size, self.window_len, 1)
        return x_out, y_out
    
    def __len__(self):
        return math.floor(len(self.x) / self.frame_size)
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(LSTMNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rec = LSTM(input_size, hidden_size, num_layers, bias=bias)
        self.lin = Linear(hidden_size, 1, bias=bias)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        rec_out, (self.hn, self.cn) = self.rec(x, (self.hn, self.cn))  
        out = self.lin(rec_out)
        out = out.permute(1, 0, 2)
        return out
    
    def reset_hiddens(self, batch_size, device):
        self.hn = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)
        self.cn = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)

    def train_epoch(self, dataset:FxDataset, loss_function, optimizer:torch.optim.Optimizer, device, shuffle=False):
        if shuffle == True:
            shuffle = torch.randperm(len(dataset))
        else:
            shuffle = range(len(dataset))
        epoch_loss = 0
        for i in range(len(dataset)):
            x, y = dataset[shuffle[i]]
            # Use the first batch to init parameters
            if i == 0:
                batch_size = x.shape[0]
                self.reset_hiddens(batch_size, device)
                y_pred = self.forward(x)
                continue
            batch_size = x.shape[0]
            optimizer.zero_grad()
            self.reset_hiddens(batch_size, device)
            y_pred = self.forward(x)
            loss = loss_function(y, y_pred)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss
            print("Frame {}/{}: {:.2%}".format(i, len(dataset), i/len(dataset)), end='\r')
        epoch_loss = epoch_loss / (i + 1)
        return epoch_loss
    
    def valid_epoch(self, dataset:FxDataset, loss_function, device):
        with torch.no_grad():
            epoch_loss = 0
            for i in range(len(dataset)):
                x, y = dataset[i]
                batch_size = x.shape[0]
                self.reset_hiddens(batch_size, device)
                y_pred = self.forward(x)
                loss = loss_function(y, y_pred)
                epoch_loss = epoch_loss + loss
            epoch_loss = epoch_loss / (i + 1)
            return epoch_loss