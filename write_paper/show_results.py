import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
import SimpleITK as sitk
from skimage import transform
from datasets.augment import resize_point

if __name__ == '__main__':
    data_dir = '/home/pangshumao/data/Spine_Localization_PIL'
    single_identifier = 'segmentation_DeepLabv3_plus_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_200'
    multi_task_60_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_60.0_batch_size_4_epochs_200'
    multi_task_w_60_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_60.0_use_weak_label_batch_size_4_epochs_200'

    fold_ind = 4
    case_ind = 58

    out_dir = os.path.join(data_dir, 'figures')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    structure_names = ['L1', 'L2', 'L3', 'L4', 'L5',
                       'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    img_size = (780, 780)
    colors = ['black', 'blue', 'lemonchiffon', 'Lime', 'yellow', 'brown',
              'purple', 'rosybrown', 'red', 'goldenrod', 'tan']
    cmap = mpl.colors.ListedColormap(colors)

    mr_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'MR', 'Case' + str(case_ind) + '.nii.gz'))
    mr = mr_nii.get_data()[:, :, 0]
    mr = np.rot90(mr)
    # mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)

    mask_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'processed_mask', 'Case' + str(case_ind) + '.nii.gz'))
    mask = mask_nii.get_data()[:, :, 0]
    mask = np.rot90(mask)
    # mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant', anti_aliasing=False)

    single_seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), single_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    single_seg = single_seg_nii.get_data()[:, :, 0]
    single_seg = np.rot90(single_seg)
    # single_seg = transform.resize(single_seg.astype(np.float32), img_size, order=0, mode='constant',
    #                                      anti_aliasing=False)

    multi_task_60_seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), multi_task_60_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    multi_task_60_seg = multi_task_60_seg_nii.get_data()[:, :, 0]
    multi_task_60_seg = np.rot90(multi_task_60_seg)
    # multi_task_60_seg = transform.resize(multi_task_60_seg.astype(np.float32), img_size, order=0, mode='constant',
    #                                      anti_aliasing=False)

    multi_task_w_60_seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), multi_task_w_60_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    multi_task_w_60_seg = multi_task_w_60_seg_nii.get_data()[:, :, 0]
    multi_task_w_60_seg = np.rot90(multi_task_w_60_seg)
    # multi_task_w_60_seg = transform.resize(multi_task_w_60_seg.astype(np.float32), img_size, order=0, mode='constant',
    #                                      anti_aliasing=False)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(20, 6),
                                        sharex=True, sharey=True)

    for aa in (ax1, ax2, ax3, ax4, ax5):
        aa.set_axis_off()

    ax1.imshow(mr, cmap='gray')
    ax1.set_title('MR')

    ax2.imshow(mask, cmap=cmap)
    ax2.set_title('Mask')

    ax3.imshow(single_seg, cmap=cmap)
    ax3.set_title('Single-task')

    ax4.imshow(multi_task_60_seg, cmap=cmap)
    ax4.set_title('Multi-task-60')

    ax5.imshow(multi_task_w_60_seg, cmap=cmap)
    ax5.set_title('Multi-task-w-60')

    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'fold' + str(fold_ind) + '_' + 'case_' + str(case_ind) + '.tiff'), dpi=300)
    plt.show()

