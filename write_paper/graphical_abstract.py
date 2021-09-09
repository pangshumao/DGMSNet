import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
import SimpleITK as sitk
from skimage import transform
from datasets.augment import resize_point
from datasets.utils import generate_heatmaps

if __name__ == '__main__':
    data_dir = '/home/pangshumao/data/Spine_Localization_PIL'
    out_dir = os.path.join(data_dir, 'figures')
    identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv'
    # fold_ind = 3
    # case_ind = 1

    fold_ind = 2
    case_ind = 180
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

    seg_nii = nib.load(os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier,
                                    'seg_' + str(case_ind) + '.nii.gz'))
    seg = seg_nii.get_data()[:, :, 0]
    seg = np.rot90(seg)
    seg = transform.resize(seg.astype(np.float32), img_size, order=0, mode='constant',
                           anti_aliasing=False)

    mask_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'processed_mask', 'Case' + str(case_ind) + '.nii.gz'))
    mask = mask_nii.get_data()[:, :, 0]
    mask = np.rot90(mask)
    mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant', anti_aliasing=False)

    kp_npz = np.load(os.path.join(data_dir, 'in', 'keypoints', 'Case' + str(case_ind) + '.npz'))
    coords = kp_npz['coords']  # (10, 2)
    original_img_size = kp_npz['img_size']  # (2, ), [height, width]
    pixelspacing = kp_npz['pixelspacing']  # (2, ), [pixel_size_w, pixel_size_h]
    pixelspacing[0] = pixelspacing[0] * original_img_size[1] / img_size[1]
    pixelspacing[1] = pixelspacing[1] * original_img_size[0] / img_size[0]
    coords = resize_point(coords, in_img_size=original_img_size, out_img_size=img_size)

    heatmaps = generate_heatmaps(mr, pixelspacing, coords, sigma=5.0)
    merge_heatmap = heatmaps.sum(axis=0)\

    fig = plt.gcf()
    fig.set_size_inches(img_size[1] / 300., img_size[0] / 300.0)  # for dpi=300, output = w * h pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(mr, cmap='gray')
    plt.margins(0, 0)
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'mr_case' + str(case_ind) + '.tiff'), dpi=300)

    fig = plt.gcf()
    fig.set_size_inches(img_size[1] / 300., img_size[0] / 300.0)  # for dpi=300, output = w * h pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(seg, cmap=cmap)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, 'seg_case' + str(case_ind) + '.tiff'), dpi=300)

    fig = plt.gcf()
    fig.set_size_inches(img_size[1] / 300., img_size[0] / 300.0)  # for dpi=300, output = w * h pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(mask, cmap=cmap)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, 'mask_case' + str(case_ind) + '.tiff'), dpi=300)

    fig = plt.gcf()
    fig.set_size_inches(img_size[1] / 300., img_size[0] / 300.0)  # for dpi=300, output = w * h pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(merge_heatmap)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, 'heatmaps_case' + str(case_ind) + '.tiff'), dpi=300)





