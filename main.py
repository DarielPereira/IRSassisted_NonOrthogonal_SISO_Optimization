"""
This code simulates the uplink of multi-user SISO systems assisted by Reconfigurable Intelligent Surface (RIS). It
evaluates the supervised and self-supervised (goal-oriented learning/objective-driven unsupervised learning/
direct metric optimization) learning approaches for optimizing the RIS configuration to maximize the sum-rate. Comparison
with the optimal scheme is presented.
"""


import os

# Set the environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.io
from functionsNN import NN_supervised
import torch as th

from functionsChannels import ChannelsUL

# Parameters for figures
fs = 12   # fontsize
lw = 1.5  # linewidth
ms = 8    # markersize
Testing = True

# Scenario parameters
Numsim = 10          # number of simulations
NumSetups = 10        # number of setups
M = 32              # number of RIS elements
K = 5               # number of users
PmaxdBm = 10        # Pmax (in dBm) of each user
Pmax = 10**(PmaxdBm/10)  # Pmax
B = 20              # Bandwidth MHz
NF = 0              # Noise Factor in dBs at the BS
noiseVariancedBm = -174 + 10 * np.log10(B * 10**6) + NF
sigma2n = 10**(noiseVariancedBm / 10)  # noise variance of the AWGN at the BS

# Channel parameters
channelparams = {
    'blocked': 1,  # Set to 1 if all direct channels are blocked
    'RiceFactorBS': 5,  # Rician factor for the channel btw RIS and BS
    'RiceFactorUE': 0,  # Rician factor for the channel btw RIS and UEs
    'pl_0': -28,  # Path loss at a reference distance (d_0)
    'alpha_RIS': 2,  # Path loss exponent for the RIS links
    'alpha_direct': 4  # Path loss exponent for the direct links
}

if Testing:
    seed = 0
    np.random.seed(seed)

# Placeholder for sum rates
sum_rate = np.zeros(Numsim*NumSetups)
sum_rate_rnd = np.zeros(Numsim*NumSetups)
sum_rateNN_supervised = np.zeros(Numsim*NumSetups)
sum_rateNN_self_supervised = np.zeros(Numsim*NumSetups)
sum_rateNN_supervised_self_supervised = np.zeros(Numsim*NumSetups)

# Positions of the users, RIS, and BS (units in meters)
sqr_size = 50  # we consider a square of size sqr_size
indices = np.linspace(0, 1, K)
PosBS_XYZ = np.array([0, 0, 3])  # BS coordinates (fixed)
PosRIS_XYZ = np.array([5, 0, 6])  # RIS coordinates (fixed)

for setup in range(NumSetups):
    print(f"Setup {setup + 1}")

    PosUE_XYZ = np.column_stack((10 + (sqr_size - 10) * np.random.rand(K),
                             10 * (np.random.rand(K)),
                             np.ones(K)))  # UEs random positions

    # Computations begin here
    for sim in range(Numsim):
        print(f"Simulation {sim + 1}")
        # Scenario generation
        hd, Hk, hBS = ChannelsUL(K, M, PosBS_XYZ, PosUE_XYZ, PosRIS_XYZ, channelparams)
        # we do not use hd as we assume blocked direct links

        # # Load the variable from the .mat file
        # mat_data = scipy.io.loadmat('data.mat')
        # PosUE_XYZ = mat_data['PosUE_XYZ']
        # hd = mat_data['hd']
        # Hk = mat_data['Hk']
        # hBS = mat_data['hBS']

        # Build auxiliary matrix H
        H = np.zeros((M, M), dtype=complex)
        for k in range(K):
            h_hat_k = np.conj(hBS) * (Hk[:, k]).reshape(-1, 1)
            H += Pmax * np.outer(h_hat_k, h_hat_k.conj())

        # Solve convex SDR problem
        W = cp.Variable((M, M), hermitian=True)
        objective = cp.Maximize(cp.real(cp.trace(H @ W)))
        constraints = [cp.diag(W) == np.ones(M), W >> 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        # Extract RIS solution from W
        eig_val, eig_vec = np.linalg.eigh(W.value)
        max_idx = np.argmax(eig_val)
        w_opt = np.sqrt(eig_val[max_idx]) * eig_vec[:, max_idx]
        w_opt = np.conj(w_opt) / np.abs(w_opt)
        Theta = np.diag(w_opt)  # This is the diagonal RIS solution that maximizes the sum rate
        sum_rate[setup*Numsim + sim] = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ Theta @ Hk)**2 / sigma2n)

        # Random RIS solution (just for comparison)
        Thetarnd = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(M)))
        sum_rate_rnd[setup*Numsim + sim] = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ Thetarnd @ Hk)**2 / sigma2n)

        # NN models
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        input = th.tensor(H, dtype=th.cfloat).reshape(1, -1).to(device)

        # load supervised model
        model_superv = NN_supervised(2*input.numel(), 2*w_opt.shape[0]).to(device)
        model_superv.load_model('./TrainingData/Model_M_32_K_5_setups_1_realiz_200000_supervised_Epochs_5_NN1_0.0001.pt')
        model_superv.eval()
        with th.no_grad():
            y = model_superv(input).cpu().numpy()
        ThetaNN = np.diag(y[0, :M] + 1j * y[0, M:])
        sum_rateNN_supervised[setup*Numsim + sim] = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ ThetaNN @ Hk)**2 / sigma2n)

        # load self-supervised model
        model_self_superv = NN_supervised(2*input.numel(), 2*M).to(device)
        model_self_superv.load_model('./TrainingData/Model_M_32_K_5_setups_1_realiz_200000_self_supervised_Epochs_5_NN1_0.0001.pt')
        model_self_superv.eval()
        with th.no_grad():
            y = model_self_superv(input).cpu().numpy()
        ThetaNN = np.diag(y[0, :M] + 1j * y[0, M:])
        sum_rateNN_self_supervised[setup*Numsim + sim] = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ ThetaNN @ Hk)**2 / sigma2n)

        # load supervised and self-supervised model
        model_superv_self_superv = NN_supervised(2*input.numel(), 2*w_opt.shape[0]).to(device)
        model_superv_self_superv.load_model('./TrainingData/Model_M_32_K_5_setups_1_realiz_200000_self_supervised_Epochs_5_NN1_1e-05.pt')
        model_superv_self_superv.eval()
        with th.no_grad():
            y = model_superv_self_superv(input).cpu().numpy()
        ThetaNN = np.diag(y[0, :M] + 1j * y[0, M:])
        sum_rateNN_supervised_self_supervised[setup*Numsim + sim] = np.log2(1 + Pmax * np.linalg.norm(hBS.conj().T @ ThetaNN @ Hk)**2 / sigma2n)


sum_rate_ave = np.mean(sum_rate)
sum_rate_rnd_ave = np.mean(sum_rate_rnd)
sum_rateNN_supervised_ave = np.mean(sum_rateNN_supervised)
sum_rateNN_self_supervised_ave = np.mean(sum_rateNN_self_supervised)
sum_rateNN_supervised_self_supervised_ave = np.mean(sum_rateNN_supervised_self_supervised)
print(f'Average sum rate SDR: {sum_rate_ave:.2f} b/s/Hz')
print(f'Average sum rate Random RIS: {sum_rate_rnd_ave:.2f} b/s/Hz')
print(f'Average sum rate NN supervised: {sum_rateNN_supervised_ave:.2f} b/s/Hz')
print(f'Average sum rate NN self-supervised: {sum_rateNN_self_supervised_ave:.2f} b/s/Hz')
print(f'Average sum rate NN supervised and self-supervised: {sum_rateNN_supervised_self_supervised_ave:.2f} b/s/Hz')

plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')

# Plotting the results
fig, ax = plt.subplots()
plt.grid(visible = True, linestyle='--')

plt.plot(sum_rate, 'r', markersize=ms, linewidth=lw, label='SDR')
plt.plot(sum_rate_rnd, 'b', markersize=ms, linewidth=lw, label='Random RIS')
plt.plot(sum_rateNN_supervised, 'g', markersize=ms, linewidth=lw, label='NN supervised')
plt.plot(sum_rateNN_self_supervised, 'm', markersize=ms, linewidth=lw, label='NN self-supervised')
plt.plot(sum_rateNN_supervised_self_supervised, 'c', markersize=ms, linewidth=lw, label='NN supervised and self-supervised')
plt.legend(loc='best')
plt.xlabel('realization')
plt.ylabel('sum-rate (b/s/Hz)')
plt.show()

image_format = 'png' # e.g .png, .svg, etc.
image_name = 'IRS_NNs.png'
fig.savefig(image_name, format=image_format, dpi=400)

# Ave. values bar graph
fig, ax = plt.subplots(figsize=(10, 6))  # Ajusta el tamaño de la figura
plt.grid(visible=True, linestyle='--')

methods = ['SDR', 'Random RIS', 'superv.', 'self-superv', 'superv+self-super']
values = np.round([sum_rate_ave, sum_rate_rnd_ave, sum_rateNN_supervised_ave, sum_rateNN_self_supervised_ave,
                   sum_rateNN_supervised_self_supervised_ave], 2)

# Create bar plot
bars = plt.bar(methods, values, color=['r', 'b', 'g', 'm', 'c'])
plt.xlabel('Methods')
plt.ylabel('Average sum-rate (b/s/Hz)')

# Annotate values on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value), ha='center', fontsize=12)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.xticks(rotation=70, fontsize=12)
plt.ylim(0, 18)
plt.subplots_adjust(bottom=0.3)  # Ajusta el margen inferior

# Show the plot
plt.show()

# Save the plot
image_format = 'png'  # e.g .png, .svg, etc.
image_name = 'IRS_NNs_ave.png'
fig.savefig(image_name, format=image_format, dpi=400)