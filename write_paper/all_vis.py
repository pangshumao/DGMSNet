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

def crop_coords(coords, img_size):
    '''

    :param coords: a numpy array with shape of (point_num, 2)
    :param img_size:
    :return:
    '''
    h, w = img_size
    start_w = w//4
    x = coords[:, 0]
    y = coords[:, 1]

    x = x - start_w
    crop_coord = np.stack([x, y], axis=1)
    return crop_coord

if __name__ == '__main__':
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL'
    out_dir = os.path.join(data_dir, 'figures')
    identifiers = [
        'segmentation_DeepLabv3_plus_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_30.0_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_50.0_use_weak_label_batch_size_4_epochs_200',
        'segmentation_DeepLabv3_plus_gcn_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_50_beta_0.3_gcn_num_2_pre_trained',
        'MRLN_Adam_lr_0.005_epoch_400_batch_size_4_beta_40.0',
        'Vgg_swdn_Adam_lr_0.0001_epoch_100_batch_size_4',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv']

    titles = ['DGMSNet-w/o-D',
              'DGMSNet-w/o-W',
              'DGMSNet-w/o' + '\n' + '-DGLF',
              'GCSN',
              'MRLN',
              'SWDN',
              'DGSSNet']

    structure_names = ['L1', 'L2', 'L3', 'L4', 'L5',
                       'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    img_size = (780, 780)

    colors = ['black', 'blue', 'lemonchiffon', 'Lime', 'yellow', 'brown',
              'purple', 'rosybrown', 'deeppink', 'goldenrod', 'tan']
    cmap = mpl.colors.ListedColormap(colors)

    fontsize = 13

    masks = []
    coords = []
    # fold1, case91
    # fold2, case9
    # fold4, case164
    for fold_ind in [4]:
        for case_ind in [164]:
            mr_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'MR', 'Case' + str(case_ind) + '.nii.gz'))
            mr = mr_nii.get_data()[:, :, 0]
            mr = np.rot90(mr)
            original_img_size = mr.shape
            mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
            mr = crop(mr)

            mask_nii = nib.load(
                os.path.join(data_dir, 'in', 'nii', 'processed_mask', 'Case' + str(case_ind) + '.nii.gz'))
            mask = mask_nii.get_data()[:, :, 0]
            mask = np.rot90(mask)
            mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant',
                                    anti_aliasing=False)
            mask = crop(mask)

            kp_npz = np.load(os.path.join(data_dir, 'in', 'keypoints', 'Case' + str(case_ind) + '.npz'))
            gt_coord = kp_npz['coords']  # (10, 2)
            gt_coord = resize_point(gt_coord, in_img_size=original_img_size, out_img_size=img_size)

            for identifier in identifiers:
                seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier,
                                                       'seg_' + str(case_ind) + '.nii.gz'))
                seg = seg_nii.get_data()[:, :, 0]
                seg = np.rot90(seg)
                seg = transform.resize(seg.astype(np.float32), img_size, order=0, mode='constant',
                                              anti_aliasing=False)
                seg = crop(seg)
                masks.append(seg)

                kp_path = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier,
                                               'keypoints_case' + str(case_ind) + '.npz')
                if os.path.exists(kp_path):
                    kp_data = np.load(kp_path)
                    coord = kp_data['pred_coords']
                    coord = resize_point(coord, in_img_size=original_img_size, out_img_size=img_size)
                    coord = crop_coords(coord, img_size)
                    coords.append(coord)

            masks.append(mask)
            coords.append(gt_coord)

            # fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=1, ncols=8, figsize=(12.34, 3.5),
            #                                                    sharex=True, sharey=True)
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=9, figsize=(13.88, 3.5),
                                                                         sharex=True, sharey=True)

            for aa in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
                aa.set_axis_off()

            ax1.imshow(mr, cmap='gray')
            ax1.set_title('MR', size=fontsize)

            ax2.imshow(masks[-1], cmap)
            ax2.set_title('Ground truth', size=fontsize)

            count = 0
            for ax in [ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
                ax.imshow(masks[count], cmap=cmap)
                ax.set_title(titles[count], size=fontsize)
                count += 1


            plt.subplots_adjust(left=0.0, right=1.0, top=1, bottom=-0.12, wspace=0.02)
            plt.savefig(os.path.join(out_dir, 'fold' + str(fold_ind) + '_' + 'case_' + str(case_ind) + '.tiff'), dpi=300)
            plt.show()

