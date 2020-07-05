#!/usr/bin/env python3

"""
Functions to implement tests for introgression using various summary statistics.
"""

import numpy as np

def abba_baba_stats(G):
    """
    Population-based estimate of summary statistic for estimating hybridizaiton/introgression.

    Returns a dictionary with the name of each statistic as keys and the associated summary
    statistic as the value.
    """
    p_1 = [np.sum(G[0:5,j]) / 5.0 for j in range(G.shape[1])]
    p_2 = [np.sum(G[5:10,j]) / 5.0 for j in range(G.shape[1])]
    p_3 = [np.sum(G[10:15,j]) / 5.0 for j in range(G.shape[1])]
    out = [np.sum(G[15:20,j]) / 5.0 for j in range(G.shape[1])]

    abba = np.array([(1.0 - p_1[i]) * p_2[i] * p_3[i] * (1.0 - out[i]) for i in range(len(p_1))])
    baba = np.array([p_1[i] * (1.0 - p_2[i]) * p_3[i] * (1.0 - out[i]) for i in range(len(p_1))])
    bbaa = np.array([p_1[i] * p_2[i] * (1.0 - p_3[i]) * (1.0 - out[i]) for i in range(len(p_1))])

    # Implementing the dynamic denominator for f_d from Martin et al. (MBE)
    abba2 = np.array([(1.0 - p_1[i]) * p_3[i] * p_3[i] * (1.0 - out[i]) for i in range(len(p_1))])
    baba2 = np.array([p_1[i] * (1.0 - p_3[i]) * p_3[i] * (1.0 - out[i]) for i in range(len(p_1))])

    res = {
        "D"     : np.sum(abba - baba) / np.sum(abba + baba),
        "f_hom" : np.sum(abba - baba) / np.sum(abba2 - baba2),
        "D_p"   : np.abs(np.sum(abba - baba)) / np.sum(abba + baba + bbaa)
    }
    return res
