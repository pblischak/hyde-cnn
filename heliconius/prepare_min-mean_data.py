#!/usr/bin/env python3

"""
<< prepare_data.py >>


Arguments
---------



Output
------


"""

import numpy as np
from random import shuffle
from sys import exit

if __name__ == '__main__':
    print("Preparing data for min and mean pairwise coalescent times:")
    no_hybridization_min = np.concatenate(
            [np.load("heliconius_no_hybridization_{}.npz".format(i))['min'] for i in [1,2,3,4]],
            axis=0
    )
    no_hybridization_mean = np.concatenate(
            [np.load("heliconius_no_hybridization_{}.npz".format(i))['mean'] for i in [1,2,3,4]],
            axis=0
    )
    no_hybridization_norm  = np.zeros((no_hybridization_min.shape[0],no_hybridization_min.shape[1],no_hybridization_min.shape[2],2))
    admixture_min = np.concatenate(
            [np.load("heliconius_admixture_{}.npz".format(i))['min'] for i in [1,2,3,4]],
            axis=0
    )
    admixture_mean = np.concatenate(
            [np.load("heliconius_admixture_{}.npz".format(i))['mean'] for i in [1,2,3,4]],
            axis=0
    )
    admixture_norm = np.zeros((admixture_min.shape[0],admixture_min.shape[1],admixture_min.shape[2],2))
    gene_flow_min = np.concatenate(
            [np.load("heliconius_gene_flow_{}.npz".format(i))['min'] for i in [1,2,3,4]],
            axis=0
    )
    gene_flow_mean = np.concatenate(
            [np.load("heliconius_gene_flow_{}.npz".format(i))['mean'] for i in [1,2,3,4]],
            axis=0
    )
    gene_flow_norm = np.zeros((gene_flow_min.shape[0],gene_flow_min.shape[1],gene_flow_min.shape[2],2))

    for i in range(no_hybridization.shape[0]):
        no_hybridization_norm[i,:,:,0] = no_hybridization_min[i,:,:] / np.max(no_hybridization_min[i,:,:])
        no_hybridization_norm[i,:,:,1] = no_hybridization_mean[i,:,:] / np.max(no_hybridization_mean[i,:,:])
        admixture_norm[i,:,:,0]        = admixture_min[i,:,:] / np.max(admixture_min[i,:,:])
        admixture_norm[i,:,:,1]        = admixture_mean[i,:,:] / np.max(admixture_mean[i,:,:])
        gene_flow_norm[i,:,:,0]        = gene_flow_min[i,:,:] / np.max(gene_flow_min[i,:,:])
        gene_flow_norm[i,:,:,1]        = gene_flow_mean[i,:,:] / np.max(gene_flow_mean[i,:,:])

    no_hybridization_shf = list(range(20000))
    shuffle(no_hybridization_shf)
    admixture_shf = list(range(20000))
    shuffle(admixture_shf)
    gene_flow_shf = list(range(20000))
    shuffle(gene_flow_shf)

    X_train_tmp = np.concatenate(
        (no_hybridization_norm[no_hybridization_shf[:15000],:,:,:],
         admixture_norm[admixture_shf[:15000],:,:,:],
         gene_flow_norm[gene_flow_shf[:15000],:,:,:]),
         axis=0
    )
    y_train_tmp = np.stack(
        (np.repeat((1,0,0), 15000),
         np.repeat((0,1,0), 15000),
         np.repeat((0,0,1), 15000)),
         axis=1
    )
    train_shf = list(range(45000))
    shuffle(train_shf)

    X_val_tmp = np.concatenate(
        (no_hybridization_norm[no_hybridization_shf[15000:17500],:,:,:],
         admixture_norm[admixture_shf[15000:17500],:,:,:],
         gene_flow_norm[gene_flow_shf[15000:17500],:,:,:]),
         axis=0
    )
    y_val_tmp = np.stack(
        (np.repeat((1,0,0), 2500),
         np.repeat((0,1,0), 2500),
         np.repeat((0,0,1), 2500)),
         axis=1
    )
    val_shf = list(range(7500))
    shuffle(val_shf)

    X_test_tmp = np.concatenate(
        (no_hybridization_norm[no_hybridization_shf[17500:],:,:,:],
         admixture_norm[admixture_shf[17500:],:,:,:],
         gene_flow_norm[gene_flow_shf[17500:],:,:,:]),
         axis=0
    )
    y_test_tmp = np.stack(
        (np.repeat((1,0,0), 2500),
         np.repeat((0,1,0), 2500),
         np.repeat((0,0,1), 2500)),
         axis=1
    )
    test_shf = list(range(7500))
    shuffle(test_shf)

    np.savez_compressed(
        'heliconius_min-mean_data.npz',
        xtrain=X_train_tmp[train_shf,:,:,:],
        xval=X_val_tmp[val_shf,:,:,:],
        xtest=X_test_tmp[test_shf,:,:,:],
        ytrain=y_train_tmp[train_shf,:],
        yval=y_val_tmp[val_shf,:],
        ytest=y_test_tmp[test_shf,:]
    )
