import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
import SimpleITK as sitk
from skimage import transform
from datasets.augment import resize_point

def crop(image):
    h, w = image.shape
    start_w = w//4
    end_w = int(w * 0.75)
    crop_image = image[:, start_w : end_w]
    return crop_image

if __name__ == '__main__':
    data_dir = '/home/pangshumao/data/Spine_Localization_PIL'
    single_identifier = 'segmentation_DeepLabv3_plus_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_200'
    two_paths_fixed_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_50.0_use_weak_label_batch_size_4_epochs_200'
    two_paths_mv_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_majority-voting'
    # two_paths_ams_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection'
    two_paths_dglf_identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv'

    fold_ind = 4
    case_ind = 39

    out_dir = os.path.join(data_dir, 'figures')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    structure_names = ['L1', 'L2', 'L3', 'L4', 'L5',
                       'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    img_size = (780, 780)

    colors = ['black', 'blue', 'lemonchiffon', 'Lime', 'yellow', 'brown',
              'purple', 'rosybrown', 'deeppink', 'goldenrod', 'tan']
    cmap = mpl.colors.ListedColormap(colors)

    mr_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'MR', 'Case' + str(case_ind) + '.nii.gz'))
    mr = mr_nii.get_data()[:, :, 0]
    mr = np.rot90(mr)
    mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
    mr = crop(mr)

    mask_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'processed_mask', 'Case' + str(case_ind) + '.nii.gz'))
    mask = mask_nii.get_data()[:, :, 0]
    mask = np.rot90(mask)
    mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant', anti_aliasing=False)
    mask = crop(mask)

    single_seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), single_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    single_seg = single_seg_nii.get_data()[:, :, 0]
    single_seg = np.rot90(single_seg)
    single_seg = transform.resize(single_seg.astype(np.float32), img_size, order=0, mode='constant',
                                         anti_aliasing=False)
    single_seg = crop(single_seg)

    two_paths_fixed_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), two_paths_fixed_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    two_paths_fixed_seg = two_paths_fixed_nii.get_data()[:, :, 0]
    two_paths_fixed_seg = np.rot90(two_paths_fixed_seg)
    two_paths_fixed_seg = transform.resize(two_paths_fixed_seg.astype(np.float32), img_size, order=0, mode='constant',
                                         anti_aliasing=False)
    two_paths_fixed_seg = crop(two_paths_fixed_seg)

    two_paths_mv_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), two_paths_mv_identifier,
                                                  'seg_' + str(case_ind) + '.nii.gz'))
    two_paths_mv_seg = two_paths_mv_nii.get_data()[:, :, 0]
    two_paths_mv_seg = np.rot90(two_paths_mv_seg)
    two_paths_mv_seg = transform.resize(two_paths_mv_seg.astype(np.float32), img_size, order=0, mode='constant',
                                         anti_aliasing=False)
    two_paths_mv_seg = crop(two_paths_mv_seg)

    # two_paths_ams_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), two_paths_ams_identifier,
    #                                          'seg_' + str(case_ind) + '.nii.gz'))
    # two_paths_ams_seg = two_paths_ams_nii.get_data()[:, :, 0]
    # two_paths_ams_seg = np.rot90(two_paths_ams_seg)
    # two_paths_ams_seg = transform.resize(two_paths_ams_seg.astype(np.float32), img_size, order=0, mode='constant',
    #                                     anti_aliasing=False)
    # two_paths_ams_seg = crop(two_paths_ams_seg)

    two_paths_dglf_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), two_paths_dglf_identifier,
                                              'seg_' + str(case_ind) + '.nii.gz'))
    two_paths_dglf_seg = two_paths_dglf_nii.get_data()[:, :, 0]
    two_paths_dglf_seg = np.rot90(two_paths_dglf_seg)
    two_paths_dglf_seg = transform.resize(two_paths_dglf_seg.astype(np.float32), img_size, order=0, mode='constant',
                                         anti_aliasing=False)
    two_paths_dglf_seg = crop(two_paths_dglf_seg)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, figsize=(10, 3),
                                        sharex=True, sharey=True)

    for aa in (ax1, ax2, ax3, ax4, ax5, ax6):
        aa.set_axis_off()

    ax1.imshow(mr, cmap='gray')
    ax1.set_title('MR')

    ax2.imshow(mask, cmap=cmap)
    ax2.set_title('Ground truth')

    ax3.imshow(single_seg, cmap=cmap)
    ax3.set_title('Single path')

    ax4.imshow(two_paths_fixed_seg, cmap=cmap)
    ax4.set_title('Fixed loss weight')

    ax5.imshow(two_paths_mv_seg, cmap=cmap)
    ax5.set_title('MV')

    # ax6.imshow(two_paths_ams_seg, cmap=cmap)
    # ax6.set_title('AMS')

    ax6.imshow(two_paths_dglf_seg, cmap=cmap)
    ax6.set_title('Proposed')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fold' + str(fold_ind) + '_' + 'case_' + str(case_ind) + '.tiff'), dpi=300)
    # plt.show()

