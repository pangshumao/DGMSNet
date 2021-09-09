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
    fold_num = 5
    train_num = 155
    val_num = 17
    test_num = 43
    total_num = 215


    foldIndDir = '/public/pangshumao/data/Spine_Localization_PIL'

    np.random.seed(0)

    ind = np.random.permutation(total_num)
    for i in range(fold_num):
        test_ind = ind[i * test_num : (i + 1) * test_num]
        test_ind = np.sort(test_ind)
        diff_ind = np.setdiff1d(ind, test_ind)

        temp_ind = np.random.permutation(diff_ind.size)

        val_ind = diff_ind[temp_ind[0:val_num]]
        val_ind = np.sort(val_ind)
        train_ind = np.setdiff1d(diff_ind, val_ind)

        train_ind = list(train_ind + 1)
        val_ind = list(val_ind + 1)
        test_ind = list(test_ind + 1)

        np.savez(os.path.join(foldIndDir, 'split_ind_fold' + str(i + 1) + '.npz'), train_ind=train_ind,
                                                                                val_ind=val_ind, test_ind=test_ind)