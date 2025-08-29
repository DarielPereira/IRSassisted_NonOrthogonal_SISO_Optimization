"""
This code generates a dataset for training a neural network to optimize the phase shifts of a Reconfigurable Intelligent
Surface (RIS) in a wireless communication system. It simulates multiple setups and realizations of user positions and
channel conditions, computes the optimal RIS configuration using Semi-definite Relaxation (SDR), and stores the results
in a dataset for supervised or self-supervised learning.
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch as th
import math
import random
import numpy as np
import cvxpy as cp

from datetime import datetime
from functionsChannels import ChannelsUL
from functionsNN import supervised_Dataset, self_supervised_Dataset
from tqdm import tqdm

sample_mode = 'supervised'  # 'supervised' or 'self_supervised'

# Scenario parameters
numSetups = 1                  # number of setups
numRealizations = 2000000            # number of simulations

Testing = False                # set to 'True' to enable testing mode
M = 32                         # number of RIS elements
K = 5                          # number of users
PmaxdBm = 10                   # Pmax (in dBm) of each user
Pmax = 10**(PmaxdBm/10)        # Pmax
B = 20                         # Bandwidth MHz
NF = 0                         # Noise Factor in dBs at the BS
noiseVariancedBm = -174 + 10 * np.log10(B * 10**6) + NF
sigma2n = 10**(noiseVariancedBm / 10)       # Noise variance of the AWGN at the BS

# Channel parameters
channelparams = {
    'blocked': 1,               # Set to 1 if all direct channels are blocked
    'RiceFactorBS': 5,          # Rician factor for the channel btw RIS and BS
    'RiceFactorUE': 0,          # Rician factor for the channel btw RIS and UEs
    'pl_0': -28,                # Path loss at a reference distance (d_0)
    'alpha_RIS': 2,             # Path loss exponent for the RIS links
    'alpha_direct': 4           # Path loss exponent for the direct links
}

# Positions of the BS and RIS (units in meters)
sqr_size = 50  # we consider a square of size sqr_size
PosBS_XYZ = np.array([0, 0, 3])  # BS coordinates (fixed)
PosRIS_XYZ = np.array([5, 0, 6])  # RIS coordinates (fixed)

# Create the dataset to store the samples
match sample_mode:
    case 'supervised':
        dataset = supervised_Dataset()
    case 'self_supervised':
        dict_config = {'Pmax': Pmax, 'sigma2n': sigma2n}
        dataset = self_supervised_Dataset(dict_config)
    case _:
        raise ValueError('Invalid sample mode')

if Testing:
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)

# run over the different setups
for setup_idx  in range(numSetups):

    # Position of the UEs
    PosUE_XYZ = np.column_stack((10 + (sqr_size - 10) * np.random.rand(K),
                                 10 * (np.random.rand(K)),
                                 np.ones(K)))  # UEs random positions

    # run over the different realizations
    for realization_idx in tqdm(range(numRealizations), desc="Generating realizations", unit="realization"):

        # Generate the channels
        hd, Hk, hBS = ChannelsUL(K, M, PosBS_XYZ, PosUE_XYZ, PosRIS_XYZ, channelparams)


        # Build auxiliary matrix H
        H = np.zeros((M, M), dtype=complex)
        for k in range(K):
            h_hat_k = np.conj(hBS) * (Hk[:, k]).reshape(-1, 1)
            H += Pmax * np.outer(h_hat_k, h_hat_k.conj())

        # Store the sample
        match sample_mode:
            case 'supervised':

                # Solve convex SDR problem
                W = cp.Variable((M, M), hermitian=True)
                objective = cp.Maximize(cp.real(cp.trace(H @ W)))
                constraints = [cp.diag(W) == np.ones(M), W >> 0]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.SCS)

                # Extract RIS solution from W
                eig_val, eig_vec = np.linalg.eigh(W.value)
                max_idx = np.argmax(eig_val)
                Theta_diag = np.sqrt(eig_val[max_idx]) * eig_vec[:, max_idx]
                Theta_diag = np.conj(Theta_diag) / np.abs(Theta_diag)

                Theta = np.diag(Theta_diag)  # This is the diagonal RIS solution that maximizes the sum rate

                sum_rate = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ Theta @ Hk) ** 2 / sigma2n)

                dataset.add_sample(th.tensor(H, dtype=th.cfloat).flatten(), th.tensor(Theta_diag, dtype=th.cfloat).flatten())

            case 'self_supervised':
                dataset.add_sample(th.tensor(H, dtype=th.cfloat), th.tensor(hBS, dtype=th.cfloat), th.tensor(Hk, dtype=th.cfloat))


file_name = (
f'./TrainingData/Dataset_M_{M}_K_{K}_setups_{numSetups}_realiz_{numRealizations}_{sample_mode}.pt')
dataset.save(file_name)










