import os
import numpy as np
import nibabel as nib
from evaluation.utils import evaluation_metrics_each_class

if __name__ == '__main__':
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL'
    identifiers = [
        'segmentation_DeepLabv3_plus_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_200',
        'segmentation_DeepLabv3_plus_gcn_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_50_beta_0.3_gcn_num_2_pre_trained',
        'MRLN_Adam_lr_0.005_epoch_400_batch_size_4_beta_40.0',
        'Vgg_swdn_Adam_lr_0.0001_epoch_100_batch_size_4',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_30.0_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_40.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_50.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_60.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_70.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_80.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_90.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_100.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_majority-voting',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv']

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
    for identifier in identifiers:
        all_dice = []
        all_precision = []
        all_recall = []

        subject_dices = []
        subject_precisions = []
        subject_recalls = []

        subject_ver_dices = []
        subject_ver_precisions = []
        subject_ver_recalls = []

        subject_ivd_dices = []
        subject_ivd_precisions = []
        subject_ivd_recalls = []
        for fold_ind in range(1, 6):
            seg_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier)
            fold_ind_data = np.load(os.path.join(data_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
            train_ind = fold_ind_data['train_ind']
            val_ind = fold_ind_data['val_ind']
            test_ind = fold_ind_data['test_ind']
            for case_ind in test_ind:
                # print('processing fold%s, case%d' % (fold_ind, case_ind))
                mask_nii = nib.load(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
                mask = mask_nii.get_data()  # [h, w, 1]
                mask = mask[:, :, 0]

                seg_nii = nib.load(os.path.join(seg_dir, 'seg_' + str(case_ind) + '.nii.gz'))
                seg = seg_nii.get_data()  # [h, w, 1]
                seg = seg[:, :, 0]

                dice, precision, recall = evaluation_metrics_each_class(seg, mask, class_num=class_num, empty_value=-1.0)
                all_dice.append(dice)
                all_precision.append(precision)
                all_recall.append(recall)

                dice = np.where(dice == -1.0, np.nan, dice)
                # print(np.nanmean(dice[1:]))
                precision = np.where(precision == -1.0, np.nan, precision)
                recall = np.where(recall == -1.0, np.nan, recall)

                subject_dices.append(np.nanmean(dice[1:]))
                subject_precisions.append(np.nanmean(precision[1:]))
                subject_recalls.append(np.nanmean(recall[1:]))

                subject_ver_dices.append(np.nanmean(dice[1:6]))
                subject_ver_precisions.append(np.nanmean(precision[1:6]))
                subject_ver_recalls.append(np.nanmean(recall[1:6]))

                subject_ivd_dices.append(np.nanmean(dice[6:]))
                subject_ivd_precisions.append(np.nanmean(precision[6:]))
                subject_ivd_recalls.append(np.nanmean(recall[6:]))

        class_dices_mean = np.nanmean([np.where(temp_dice == -1.0, np.nan, temp_dice) for temp_dice in all_dice],
                                      axis=0)
        class_precisions_mean = np.nanmean(
            [np.where(temp_precision == -1.0, np.nan, temp_precision) for temp_precision in all_precision], axis=0)
        class_recalls_mean = np.nanmean(
            [np.where(temp_recall == -1.0, np.nan, temp_recall) for temp_recall in all_recall], axis=0)

        class_dices_std = np.nanstd([np.where(temp_dice == -1.0, np.nan, temp_dice) for temp_dice in all_dice], axis=0)
        class_precisions_std = np.nanstd(
            [np.where(temp_precision == -1.0, np.nan, temp_precision) for temp_precision in all_precision], axis=0)
        class_recalls_std = np.nanstd(
            [np.where(temp_recall == -1.0, np.nan, temp_recall) for temp_recall in all_recall], axis=0)

        print('----------------------------' + identifier + '---------------------------')
        for i in range(0, class_num):
            print('%s mean dice = %.2f +- %.2f' % (
                class_name[i], class_dices_mean[i] * 100, class_dices_std[i] * 100))

        print(
            'image-level ver mean dice = %.2f +- %.2f' % (
            np.mean(subject_ver_dices) * 100, np.std(subject_ver_dices) * 100))
        print(
            'image-level ivd mean dice = %.2f +- %.2f' % (
            np.mean(subject_ivd_dices) * 100, np.std(subject_ivd_dices) * 100))
        print(
            'image-level total mean dice = %.2f +- %.2f' % (np.mean(subject_dices) * 100, np.std(subject_dices) * 100))
        print('...........................................................')

        for i in range(0, class_num):
            print('%s mean precision = %.2f +- %.2f' % (
                class_name[i], class_precisions_mean[i] * 100,
                class_precisions_std[i] * 100))

        print(
            'image-level ver mean precision = %.2f +- %.2f' % (
            np.mean(subject_ver_precisions) * 100, np.std(subject_ver_precisions) * 100))
        print(
            'image-level ivd mean precision = %.2f +- %.2f' % (
            np.mean(subject_ivd_precisions) * 100, np.std(subject_ivd_precisions) * 100))
        print('image-level total mean precision = %.2f +- %.2f' % (
        np.mean(subject_precisions) * 100, np.std(subject_precisions) * 100))
        print('...........................................................')

        for i in range(0, class_num):
            print('%s mean recall = %.2f +- %.2f' % (
                class_name[i], class_recalls_mean[i] * 100, class_recalls_std[i] * 100))

        print(
            'image-level ver mean recall = %.2f +- %.2f' % (
            np.mean(subject_ver_recalls) * 100, np.std(subject_ver_recalls) * 100))
        print(
            'image-level ivd mean recall = %.2f +- %.2f' % (
            np.mean(subject_ivd_recalls) * 100, np.std(subject_ivd_recalls) * 100))
        print('image-level total mean recall = %.2f +- %.2f' % (
        np.mean(subject_recalls) * 100, np.std(subject_recalls) * 100))

        out_dir = os.path.join(data_dir, 'result', identifier)
        if not os.path.exists(os.path.join(out_dir)):
            os.makedirs(os.path.join(out_dir))

        np.savez(os.path.join(out_dir, 'all_evaluate_metrics.npz'), all_dice=all_dice,
                 all_precision=all_precision,
                 all_recall=all_recall,
                 subject_dices=subject_dices, subject_precisions=subject_precisions, subject_recalls=subject_recalls,
                 subject_ver_dices=subject_ver_dices, subject_ver_precisions=subject_ver_precisions,
                 subject_ver_recalls=subject_ver_recalls,
                 subject_ivd_dices=subject_ivd_dices, subject_ivd_precisions=subject_ivd_precisions,
                 subject_ivd_recalls=subject_ivd_recalls,
                 class_dices_mean=class_dices_mean, class_precisions_mean=class_precisions_mean,
                 class_recalls_mean=class_recalls_mean,
                 class_dices_std=class_dices_std, class_precisions_std=class_precisions_std,
                 class_recalls_std=class_recalls_std)
