import os
import numpy as np
import nibabel as nib
from evaluation.utils import evaluation_metrics_each_class

if __name__ == '__main__':
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL/spark_challenge_test_100'
    identifiers = [
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_40.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_50.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_60.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_70.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_80.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_90.0_use_weak_label_batch_size_4_epochs_600',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_100.0_use_weak_label_batch_size_4_epochs_600']

    class_name = ['background', # 0
                  'L1', # 1
                  'L2', # 2
                  'L3', # 3
                  'L4', # 4
                  'L5', # 5
                  'L1/L2', # 6
                  'L2/L3', # 7
                  'L3/L4', # 8
                  'L4/L5', # 9
                  'L5/S1'] # 10
    class_num = len(class_name)

    mask_dir = os.path.join(data_dir, 'in', 'nii', 'processed_mask')
    pred_dists_list = []
    seg_dists_list = []
    pred_acc_list = []
    seg_acc_list = []
    for identifier in identifiers:
        for fold_ind in range(1, 2):
            out_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier)
            fold_ind_data = np.load(os.path.join(data_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
            train_ind = fold_ind_data['train_ind']
            val_ind = fold_ind_data['val_ind']
            test_ind = fold_ind_data['test_ind']
            for case_ind in test_ind:
                # print('processing fold%s, case%d' % (fold_ind, case_ind))
                np_data = np.load(os.path.join(out_dir, 'keypoints_case' + str(case_ind) + '.npz'))
                pred_dists = np_data['pred_dists']
                seg_dists = np_data['seg_dists']
                pred_acc = np_data['pred_acc']
                seg_acc = np_data['seg_acc']

                pred_dists_list.append(pred_dists)
                seg_dists_list.append(seg_dists)
                pred_acc_list.append(pred_acc)
                seg_acc_list.append(seg_acc)
    pred_dists_np = np.stack(pred_dists_list, axis=0)
    seg_dists_np = np.stack(seg_dists_list, axis=0)

    mean_pred_dist = np.mean(pred_dists_np)
    mean_seg_dist = np.mean(seg_dists_np)
    mean_pred_acc = np.mean(pred_acc_list)
    mean_seg_acc = np.mean(seg_acc_list)

    print('mean pred dist = %.2f mm, mean seg dist = %.2f mm' % (mean_pred_dist, mean_seg_dist))
    print('mean pred acc = %.4f, mean seg acc = %.4f' % (mean_pred_acc, mean_seg_acc))




