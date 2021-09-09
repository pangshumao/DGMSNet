import os
import SimpleITK as sitk
import numpy as np
from datasets.utils import generate_heatmaps
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import torch
from torch import nn
from torch.nn import functional as F
from datasets.utils import pad_image_coords_to_square
from skimage import transform
from datasets.augment import resize_point
import matplotlib as mpl

if __name__ == '__main__':
    # data_dir = '/public/pangshumao/data/Spine_Localization_PIL/spark_challenge_test_100'
    data_dir = '/public/pangshumao/data/DGMSNet/Spinal_disease_dataset'

    mr_dir = os.path.join(data_dir, 'in', 'MR')
    mask_dir = os.path.join(data_dir, 'in', 'mask')
    keypoints_dir = os.path.join(data_dir, 'in', 'keypoints')
    img_size = [512, 512]
    class_num = 11
    colors = ['black', 'blue', 'lemonchiffon', 'Lime', 'yellow', 'brown',
              'purple', 'rosybrown', 'red', 'goldenrod', 'tan']
    cmap = mpl.colors.ListedColormap(colors)
    structure_names = ['L1', 'L2', 'L3', 'L4', 'L5',
                       'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    for case_ind in [417, 428, 432, 434, 447, 460, 482]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
        image = reader.Execute()
        mr = sitk.GetArrayFromImage(image)  # 1, h, w
        mr = mr[0, :, :]

        mask_image = sitk.ReadImage(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
        mask = sitk.GetArrayFromImage(mask_image)



        pixelspacing = mask_image.GetSpacing()
        height, width = mask.shape
        # pixel_size_h = pixel_size_h * height / img_h
        # pixel_size_w = pixel_size_w * width / img_w
        coords = np.zeros((class_num - 1, 2), dtype=np.float32)
        for i in range(1, 11):
            pos = np.argwhere(mask == i)
            count = pos.shape[0]
            center_h, center_w = pos.sum(0) / count
            center_h = round(center_h)
            center_w = round(center_w)
            coords[i - 1, :] = np.array([center_w, center_h])  # x, y
        np.savez(os.path.join(keypoints_dir, 'Case' + str(case_ind) + '.npz'), coords=coords, img_size=[height, width],
                 pixelspacing=[pixelspacing[0], pixelspacing[1]])


        mask = pad_image_coords_to_square(mask)
        mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant',
                                anti_aliasing=False).astype(np.int)

        kp_npz = np.load(os.path.join(keypoints_dir, 'Case' + str(case_ind) + '.npz'))
        coords = kp_npz['coords']  # (10, 2)
        mr, coords = pad_image_coords_to_square(mr, coords)
        original_img_size = np.array(mr.shape)
        pixelspacing = kp_npz['pixelspacing']  # (2, ), [pixel_size_w, pixel_size_h]
        pixelspacing[0] = pixelspacing[0] * original_img_size[1] / img_size[1]
        pixelspacing[1] = pixelspacing[1] * original_img_size[0] / img_size[0]

        mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
        coords = resize_point(coords, in_img_size=original_img_size, out_img_size=img_size)
        point_num = coords.shape[0]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 12),
                                                      sharex=True, sharey=True)

        for aa in (ax1, ax2):
            aa.set_axis_off()

        ax1.imshow(mr, cmap='gray')
        ax1.set_title('Case%d MR' % case_ind)

        for i in range(point_num):
            x = coords[i, 0]
            y = coords[i, 1]
            ax1.scatter(x=x, y=y, c='r')
            ax1.text(x + 50, y + 10, s=structure_names[i], c='w', size=14)

        ax2.imshow(mask, cmap=cmap)
        ax2.set_title('Case%d Mask' % case_ind)

        for i in range(point_num):
            x = coords[i, 0]
            y = coords[i, 1]
            ax2.scatter(x=x, y=y, c='w')
            ax2.text(x + 50, y + 10, s=structure_names[i], c='w', size=14)


        plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, 'fold' + str(fold_ind) + '_' + 'case_' + str(case_ind) + '.tiff'), dpi=300)
        plt.show()


