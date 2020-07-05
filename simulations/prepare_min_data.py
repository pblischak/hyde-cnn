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

if __name__ == '__main__':
    print("Preparing data for mean pairwise coalescent times:")
    for cu in ['0.5', '1.0', '2.0']:
        print("  Coalescent Units {}".format(cu))
        no_hybridization       = np.load("../raw_data/no_hybridization_{}.npz".format(cu))['min']
        no_hybridization_norm  = np.zeros((no_hybridization.shape[0],no_hybridization.shape[1],no_hybridization.shape[2],1))
        hybrid_speciation      = np.load("../raw_data/hybrid_speciation_{}.npz".format(cu))['min']
        hybrid_speciation_norm = np.zeros((hybrid_speciation.shape[0],hybrid_speciation.shape[1],hybrid_speciation.shape[2],1))
        admixture              = np.load("../raw_data/admixture_{}.npz".format(cu))['min']
        admixture_norm         = np.zeros((admixture.shape[0],admixture.shape[1],admixture.shape[2],1))
        admixture_w_gflow      = np.load("../raw_data/admixture_w_gflow_{}.npz".format(cu))['min']
        admixture_w_gflow_norm = np.zeros((admixture_w_gflow.shape[0],admixture_w_gflow.shape[1],admixture_w_gflow.shape[2],1))

        for i in range(no_hybridization.shape[0]):
            no_hybridization_norm[i,:,:,0] = no_hybridization[i,:,:] / np.max(no_hybridization[i,:,:])
            hybrid_speciation_norm[i,:,:,0] = hybrid_speciation[i,:,:] / np.max(hybrid_speciation[i,:,:])
            admixture_norm[i,:,:,0] = admixture[i,:,:] / np.max(admixture[i,:,:])
            admixture_w_gflow_norm[i,:,:,0] = admixture_w_gflow[i,:,:] / np.max(admixture_w_gflow[i,:,:])

        no_hybridization_shf = list(range(20000))
        shuffle(no_hybridization_shf)
        hybrid_speciation_shf = list(range(20000))
        shuffle(hybrid_speciation_shf)
        admixture_shf = list(range(20000))
        shuffle(admixture_shf)
        admixture_w_gflow_shf = list(range(20000))
        shuffle(admixture_w_gflow_shf)

        X_train_tmp = np.concatenate(
            (no_hybridization_norm[no_hybridization_shf[:15000],:,:,:],
             hybrid_speciation_norm[hybrid_speciation_shf[:15000],:,:,:],
             admixture_norm[admixture_shf[:15000],:,:,:],
             admixture_w_gflow_norm[admixture_w_gflow_shf[:15000],:,:,:]),
             axis=0
        )
        y_train_tmp = np.stack(
            (np.repeat((1,0,0,0), 15000),
             np.repeat((0,1,0,0), 15000),
             np.repeat((0,0,1,0), 15000),
             np.repeat((0,0,0,1), 15000)),
             axis=1
        )
        train_shf = list(range(60000))
        shuffle(train_shf)

        X_val_tmp = np.concatenate(
            (no_hybridization_norm[no_hybridization_shf[15000:17500],:,:,:],
             hybrid_speciation_norm[hybrid_speciation_shf[15000:17500],:,:,:],
             admixture_norm[admixture_shf[15000:17500],:,:,:],
             admixture_w_gflow_norm[admixture_shf[15000:17500],:,:,:]),
             axis=0
        )
        y_val_tmp = np.stack(
            (np.repeat((1,0,0,0), 2500),
             np.repeat((0,1,0,0), 2500),
             np.repeat((0,0,1,0), 2500),
             np.repeat((0,0,0,1), 2500)),
             axis=1
        )
        val_shf = list(range(10000))
        shuffle(val_shf)

        X_test_tmp = np.concatenate(
            (no_hybridization_norm[no_hybridization_shf[17500:],:,:,:],
             hybrid_speciation_norm[hybrid_speciation_shf[17500:],:,:,:],
             admixture_norm[admixture_shf[17500:],:,:,:],
             admixture_w_gflow_norm[admixture_shf[17500:],:,:,:]),
             axis=0
        )
        y_test_tmp = np.stack(
            (np.repeat((1,0,0,0), 2500),
             np.repeat((0,1,0,0), 2500),
             np.repeat((0,0,1,0), 2500),
             np.repeat((0,0,0,1), 2500)),
             axis=1
        )
        test_shf = list(range(10000))
        shuffle(test_shf)

        np.savez_compressed(
            '../processed_data/hyde_cnn_min_data_{}.npz'.format(cu),
            xtrain=X_train_tmp[train_shf,:,:,:],
            xval=X_val_tmp[val_shf,:,:,:],
            xtest=X_test_tmp[test_shf,:,:,:],
            ytrain=y_train_tmp[train_shf,:],
            yval=y_val_tmp[val_shf,:],
            ytest=y_test_tmp[test_shf,:]
        )
