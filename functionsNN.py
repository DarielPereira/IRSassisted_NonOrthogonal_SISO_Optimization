"""
This module contains the implementation of the neural network models for supervised and self-supervised learning.
It includes the definition of the neural network architecture, training and validation methods, and dataset handling classes.
"""

import torch
import torch as th
from collections import deque
import random
import numpy as np
import pickle

from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from tqdm import tqdm


class NN_supervised(th.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layern_norm = th.nn.LayerNorm(input_size)
        self.linear1 = th.nn.Linear(input_size, 2 * input_size)
        self.linear2 = th.nn.Linear(2 * input_size, 4 * input_size)
        self.linear3 = th.nn.Linear(4 * input_size, output_size)
        self.relu1 = th.nn.LeakyReLU(0.01)
        self.relu2 = th.nn.LeakyReLU(0.01)

    def forward(self, H):
        x = th.cat((H.real, H.imag), dim=1).to(th.float)
        x_norm = x / th.max(th.abs(x), axis=1)[0].reshape(-1, 1)
        x1 = self.relu1(self.linear1(x_norm))
        x2 = self.relu2(self.linear2(x1))
        y = th.nn.functional.tanh(self.linear3(x2))

        # Combine real and imaginary parts for normalization
        y = torch.complex(y[:, :y.shape[1] // 2], y[:, y.shape[1] // 2:])
        # Calculate the magnitude of each complex number (element-wise)
        magnitude = torch.abs(y)
        # Normalize the complex tensor to unit norm
        y = y / magnitude
        y = th.cat((y.real, y.imag), dim=1).to(th.float)
        return y

    def model_train(self, train_loader, optimizer, loss_fn, train_mode, dict_config=None):
        self.train()
        total_loss = 0
        with (tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar):
            for batch in train_loader:
                optimizer.zero_grad()

                match train_mode:
                    case 'supervised':
                        H, Theta_diag = batch

                        # Compute the prediction
                        y = self(H)

                        # Flatten the diagonal of the Theta matrix
                        label = th.cat((Theta_diag.real, Theta_diag.imag), dim=1).to(th.float)

                        loss = loss_fn(y, label)
                        loss.backward()
                        optimizer.step()

                    case 'self_supervised':
                        H, hBs, Hk = batch

                        # Freeze the parameters of the channel
                        hBs._requires_grad = False
                        Hk._requires_grad = False

                        Pmax = dict_config['Pmax']
                        sigma2n = dict_config['sigma2n']

                        # Compute the prediction
                        y = self(H.reshape(1, -1))

                        ThetaNN = th.diag(y[0, :int(y.shape[1]/2)] + 1j * y[0, int(y.shape[1]/2):])

                        # Compute the loss
                        loss = -th.log2(1 + Pmax * th.norm(hBs[0].conj().T @ ThetaNN @ Hk[0]) ** 2 / sigma2n)

                        loss.backward()
                        optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(train_loader)

    def model_validate(self, val_loader, loss_fn):
        self.eval()
        total_loss = 0
        with th.no_grad():
            for batch in val_loader:
                H, Theta_diag = batch

                # Compute the prediction
                y = self(H)

                # Flatten the diagonal of the Theta matrix
                label = th.cat((Theta_diag.real, Theta_diag.imag), dim=1).to(th.float)

                loss = loss_fn(y, label)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, file_path):
        th.save(self.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        self.load_state_dict(th.load(file_path))
        self.eval()
        print(f'Model loaded from {file_path}')



class supervised_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def add_sample(self, H, Theta_diag):
        """Add a sample to the dataset."""
        self.data_list.append((H, Theta_diag))

    def combine(self, origen_filenames, destination_filename):
         data = th.load(origen_filenames[0])
         self.data_list = data['data_list']
         for file in origen_filenames[1:]:
            data = th.load(file)
            self.data_list += data['data_list']

         self.save(destination_filename)

    def save(self, file_path):
        th.save({'data_list': self.data_list}, file_path)
        print(f'Dataset and parameters saved to {file_path}')

    def load(self, file_path):
        data = th.load(file_path)
        self.data_list = data['data_list']
        print(f'Dataset and parameters loaded from {file_path}')


class self_supervised_Dataset(Dataset):
    def __init__(self, dict_conf = None):
        super().__init__()
        self.data_list = []
        self.dict_conf = dict_conf

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def add_sample(self, hBs, Hk, Theta_diag):
        """Add a sample to the dataset."""
        self.data_list.append((hBs, Hk, Theta_diag))

    def combine(self, origen_filenames, destination_filename):
        data = th.load(origen_filenames[0])
        self.data_list = data['data_list']
        for file in origen_filenames[1:]:
            data = th.load(file)
            self.data_list += data['data_list']

        self.save(destination_filename)

    def save(self, file_path):
        th.save({'data_list': self.data_list, 'dict_config': self.dict_conf}, file_path)
        print(f'Dataset and parameters saved to {file_path}')

    def load(self, file_path):
        data = th.load(file_path)
        self.data_list = data['data_list']
        self.dict_conf = data['dict_config']
        print(f'Dataset and parameters loaded from {file_path}')