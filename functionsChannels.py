"""
This function generates the channels for the RIS-assisted uplink scenario.
"""

import numpy as np

def ChannelsUL(K, M, PosBS_XYZ, PosUE_XYZ, PosRIS_XYZ, channelparams):
    """
    Generate channels for the RIS-assisted uplink scenario.
    The UEs and the BS are single antenna!!

    Parameters:
    K : int
        Number of users
    M : int
        Number of RIS elements
    PosBS_XYZ : np.ndarray
        Positions of the BS
    PosUE_XYZ : np.ndarray
        Positions of the UEs
    PosRIS_XYZ : np.ndarray
        Positions of the RIS
    channelparams : dict
        Dictionary with the channel parameters

    Returns:
    hd : np.ndarray
        Kx1 vector with the direct links for the K users
    Hk : np.ndarray
        MxK matrix with the channels from users to RIS
    hBS : np.ndarray
        Mx1 vector with the channel from RIS to BS
    """

    hd = np.zeros((K, 1), dtype=complex)  # Direct links
    Hk = np.zeros((M, K), dtype=complex)  # Channels from UEs to RIS
    hBS = np.zeros((M, 1), dtype=complex)  # Channel from RIS to BS

    x_rx, y_rx, z_rx = PosBS_XYZ
    x_tx, y_tx, z_tx = PosUE_XYZ.T
    x_ris, y_ris, z_ris = PosRIS_XYZ

    d_RIS_rx = np.sqrt((x_ris - x_rx)**2 + (y_ris - y_rx)**2 + (z_ris - z_rx)**2)
    d_tx_RIS = np.sqrt((x_ris - x_tx)**2 + (y_ris - y_tx)**2 + (z_ris - z_tx)**2)
    pl_tx_ris_db = np.zeros(K)
    pl_tx_ris_eff = np.zeros(K)

    # Links tx-RIS
    for k in range(K):
        pl_tx_ris_db[k] = channelparams['pl_0'] - 10 * channelparams['alpha_RIS'] * np.log10(d_tx_RIS[k])
        pl_tx_ris_eff[k] = 10**(pl_tx_ris_db[k] / 20)
        phi_AoD1 = 2 * np.pi * np.random.rand()
        a_D_r = np.exp(1j * np.pi * np.arange(M) * np.sin(phi_AoD1))
        Hk[:, k] = pl_tx_ris_eff[k] * (
            (np.sqrt(channelparams['RiceFactorUE']) / np.sqrt(channelparams['RiceFactorUE'] + 1)) * a_D_r +
            (1 / np.sqrt(channelparams['RiceFactorUE'] + 1)) * (1 / np.sqrt(2) * np.random.randn(M) + 1j * 1 / np.sqrt(2) * np.random.randn(M))
        )

    # Link RIS-rx
    pl_ris_rx_db = channelparams['pl_0'] - 10 * channelparams['alpha_RIS'] * np.log10(d_RIS_rx)
    pl_ris_rx_eff = 10**(pl_ris_rx_db / 20)
    phi_AoD1 = 2 * np.pi * np.random.rand()
    a_D_r = np.exp(1j * np.pi * np.arange(M) * np.sin(phi_AoD1))
    hBS[:, 0] = pl_ris_rx_eff * (
        (np.sqrt(channelparams['RiceFactorBS']) / np.sqrt(channelparams['RiceFactorBS'] + 1)) * a_D_r +
        (1 / np.sqrt(channelparams['RiceFactorBS'] + 1)) * (1 / np.sqrt(2) * np.random.randn(M) + 1j * 1 / np.sqrt(2) * np.random.randn(M))
    )

    # Direct Links hd(k) channel for kth UE to the BS w/o RIS
    d_tx_rx = np.sqrt((x_tx - x_rx)**2 + (y_tx - y_rx)**2)
    pl_tx_rx_db = channelparams['pl_0'] - 10 * channelparams['alpha_direct'] * np.log10(d_tx_rx)
    pl_tx_rx_eff = 10**(pl_tx_rx_db / 20)
    for k in range(K):
        hd[k, 0] = pl_tx_rx_eff[k] * (1 / np.sqrt(2) * np.random.randn() + 1j * 1 / np.sqrt(2) * np.random.randn())
        if channelparams['blocked'] == 1:
            hd[k, 0] = 0

    return hd, Hk, hBS