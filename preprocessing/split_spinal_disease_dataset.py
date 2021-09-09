import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import math
from scipy import ndimage

if __name__ == '__main__':
    fold_num = 2
    train_num = 45
    val_num = 5
    test_num = 50
    total_num = 100

    foldIndDir = '/public/pangshumao/data/Spine_Localization_PIL/spark_challenge_test_100'

    np.random.seed(0)

    # testA50 is used as training and validation set, testB50 is test dataset
    test_ind = np.linspace(467, 516, test_num).astype(np.int)
    train_ind = np.linspace(417, 466, train_num + val_num).astype(np.int)
    ind = np.random.permutation(train_ind)
    train_ind = np.sort(ind[:train_num])
    val_ind = np.sort(ind[train_num:])

    np.savez(os.path.join(foldIndDir, 'split_ind_fold1.npz'), train_ind=train_ind,
             val_ind=val_ind, test_ind=test_ind)

    # testB50 is used as training and validation set, testA50 is test dataset
    train_ind = np.linspace(467, 516, train_num + val_num).astype(np.int)
    test_ind = np.linspace(417, 466, test_num).astype(np.int)
    ind = np.random.permutation(train_ind)
    train_ind = np.sort(ind[:train_num])
    val_ind = np.sort(ind[train_num:])

    np.savez(os.path.join(foldIndDir, 'split_ind_fold2.npz'), train_ind=train_ind,
             val_ind=val_ind, test_ind=test_ind)

