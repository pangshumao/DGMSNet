import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mask_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/processed_mask'
    out_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/keypoints'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    class_num = 11
    # img_h = 880.0
    # img_w = 880.0

    file_list = os.listdir(mask_dir)
    for file_name in file_list:
        nii = nib.load(os.path.join(mask_dir, file_name))
        mask = nii.get_data()
        hdr = nii.header
        pixel_size_h, pixel_size_w = hdr['pixdim'][1:3]

        mask_slice = np.rot90(mask[:, :, 0])
        height, width = mask_slice.shape
        # pixel_size_h = pixel_size_h * height / img_h
        # pixel_size_w = pixel_size_w * width / img_w
        coords = np.zeros((class_num - 1, 2), dtype=np.float32)
        for i in range(1, 11):
            pos = np.argwhere(mask_slice == i)
            count = pos.shape[0]
            center_h, center_w = pos.sum(0) / count
            center_h = round(center_h)
            center_w = round(center_w)
            coords[i - 1, :] = np.array([center_w, center_h]) # x, y
        np.savez(os.path.join(out_dir, file_name.split('.')[0] + '.npz'), coords=coords, img_size=[height, width],
                 pixelspacing=[pixel_size_w, pixel_size_h])
        # temp = np.load(os.path.join(out_dir, file_name.split('.')[0] + '.npz'))
