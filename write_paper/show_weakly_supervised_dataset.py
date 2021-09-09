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
    out_dir = os.path.join(data_dir, 'figures')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    structure_names = ['L1', 'L2', 'L3', 'L4', 'L5',
                       'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    img_size = (780, 780)
    colors = ['black', 'blue', 'lemonchiffon', 'Lime', 'yellow', 'brown',
              'purple', 'rosybrown', 'deeppink', 'goldenrod', 'tan']

    cmap = mpl.colors.ListedColormap(colors)

    mr_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'MR', 'Case1.nii.gz'))
    mr = mr_nii.get_data()[:, :, 0]
    mr = np.rot90(mr)
    mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)

    mask_nii = nib.load(os.path.join(data_dir, 'in', 'nii', 'processed_mask', 'Case1.nii.gz'))
    mask = mask_nii.get_data()[:, :, 0]
    mask = np.rot90(mask)
    mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant', anti_aliasing=False)

    reader = sitk.ImageFileReader()
    reader.SetFileName(os.path.join(data_dir, 'in', 'weak_supervised_MR', 'Case218.dcm'))
    image = reader.Execute()
    weak_mr = sitk.GetArrayFromImage(image)  # 1, h, w
    weak_mr = weak_mr[0, :, :]
    weak_mr = transform.resize(weak_mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)

    kp_npz = np.load(os.path.join(data_dir, 'in', 'keypoints', 'Case218.npz'))
    coords = kp_npz['coords']  # (10, 2)
    original_img_size = kp_npz['img_size']  # (2, ), [height, width]
    coords = resize_point(coords, in_img_size=original_img_size, out_img_size=img_size)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 6),
                                        sharex=True, sharey=True)
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
    #                                          sharex=True, sharey=True)
    # ax1 = axs[0, 0]
    # ax2 = axs[0, 1]
    # ax3 = axs[1, 0]
    # ax4 = axs[1, 1]

    for aa in (ax1, ax2, ax3, ax4):
        aa.set_axis_off()

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    ax1.imshow(mr, cmap='gray')
    ax1.set_title('MR', font1)
    ax2.imshow(mask, cmap=cmap)

    ax2.set_title('Mask', font1)
    ax3.imshow(weak_mr, cmap='gray')
    ax3.set_title('MR', font1)
    none_image = np.zeros_like(weak_mr)
    ax4.imshow(none_image, cmap='gray')

    for i in range(coords.shape[0]):
        x = coords[i, 0]
        y = coords[i, 1]
        ax4.scatter(x=x, y=y, c='r')
        ax4.text(x + 10, y + 10, s=structure_names[i], c='w', size=14)
    ax4.set_title('Keypoints', font1)

    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'mr_mask_keypoints.tiff'), dpi=300)
    plt.show()

