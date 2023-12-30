'''
Implementation of article https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf
'''
import torch
from torch import nn
from torch.nn import LSTM, Linear, GRU
from torch.utils.data import TensorDataset
import math

class FxDataset(TensorDataset):
    def __init__(self, x, y, window_len, batch_size, shuffle=False):
        super(TensorDataset, self).__init__()
        len_index = len(x) - (len(x) % (window_len * batch_size))
        self.x = x[:len_index].reshape(-1, window_len, 1)
        self.y = y[:len_index].reshape(-1, window_len, 1)
        if shuffle == True:
            index = torch.randperm(len(self.x))
            self.x = self.x[index]
            self.y = self.y[index]
        self.batch_size = batch_size

    def __getitem__(self, index):
        x_out = self.x[index*self.batch_size : (index+1)*self.batch_size]
        y_out = self.y[index*self.batch_size : (index+1)*self.batch_size]
        return x_out, y_out
    
    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))

class RnnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(RnnNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # self.rec = LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=True)
        self.rec = GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=True)
        self.lin = Linear(hidden_size, 1, bias=bias)

    def forward(self, x):
        # rec_out, (self.hn, self.cn) = self.rec(x, (self.hn, self.cn))
        rec_out, self.hn,  = self.rec(x, self.hn)
        out = self.lin(rec_out)
        out += x
        return out
    
    def reset_hiddens(self, batch_size, device):
        self.hn = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)
        # self.cn = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)

    def detach_hidden(self):
        if self.hn.__class__ == tuple:
            self.hn = tuple([h.clone().detach() for h in self.hn])
            # self.cn = tuple([c.clone().detach() for c in self.cn])
        else:
            self.hn = self.hn.clone().detach()
            # self.cn = self.cn.clone().detach()

    def train_epoch(self, dataset:FxDataset, loss_function, optimizer:torch.optim.Optimizer, device, init_len=200, up_fr=1000):
        epoch_loss = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            batch_size = x.shape[0]
            self.reset_hiddens(batch_size, device)

            # Init parameters
            self.forward(x[:, :init_len, :])
            x = x[:, init_len:, :]
            y = y[:, init_len:, :]
            optimizer.zero_grad()

            batch_loss = 0
            for j in range(math.ceil(x.shape[1] / up_fr)):
                y_pred = self.forward(x[:, j*up_fr : (j+1)*up_fr, :])
                loss = loss_function(y[:, j*up_fr : (j+1)*up_fr, :], y_pred)
                loss.backward()
                optimizer.step()
                self.detach_hidden()
                optimizer.zero_grad()
                batch_loss += loss
            batch_loss /= (j + 1)
            epoch_loss += batch_loss
            print("Batch {}/{}: {:.2%}".format(i, len(dataset), i/len(dataset)), end='\r')
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