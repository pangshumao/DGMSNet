import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tarfile
import SimpleITK as sitk
import pydicom

def change_mask():
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/original_mask'
    out_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/processed_mask'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_list = os.listdir(data_dir)
    count = 0
    for file_name in file_list:
        count += 1
        print(count)
        nii = nib.load(os.path.join(data_dir, file_name))
        mask = nii.get_data()

        ver_mask = np.where(mask <= 6, mask, 0) # [0, 1, 2, 3, 4, 5, 6]
        ver_mask = np.where(ver_mask == 1, 0, ver_mask) # [0, 2, 3, 4, 5, 6]
        ver_mask = ver_mask - np.where(ver_mask > 0, 1, 0) # [0, 1, 2, 3, 4, 5]
        mask_L1 = np.where(ver_mask == 5, 1, 0)
        mask_L2 = np.where(ver_mask == 4, 1, 0) * 2
        mask_L3 = np.where(ver_mask == 3, 1, 0) * 3
        mask_L4 = np.where(ver_mask == 2, 1, 0) * 4
        mask_L5 = np.where(ver_mask == 1, 1, 0) * 5
        ver_mask = mask_L1 + mask_L2 + mask_L3 + mask_L4 + mask_L5


        disc_binary_mask = np.where(mask <= 15, 1, 0) * np.where(mask >= 11, 1, 0)
        disc_mask = mask * disc_binary_mask # [11, 12, 13, 14, 15]
        mask_disc1 = np.where(disc_mask == 15, 1, 0) * 6
        mask_disc2 = np.where(disc_mask == 14, 1, 0) * 7
        mask_disc3 = np.where(disc_mask == 13, 1, 0) * 8
        mask_disc4 = np.where(disc_mask == 12, 1, 0) * 9
        mask_disc5 = np.where(disc_mask == 11, 1, 0) * 10
        disc_mask = mask_disc1 + mask_disc2 + mask_disc3 + mask_disc4 + mask_disc5

        mask = ver_mask + disc_mask
        out_mask_nii = nib.Nifti1Image(mask.astype(np.int16), affine=nii.affine)
        nib.save(out_mask_nii, os.path.join(out_dir, file_name))

def show_mask():
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/processed_mask'

    file_list = os.listdir(data_dir)
    count = 0
    for file_name in file_list:
        count += 1
        print(count)
        nii = nib.load(os.path.join(data_dir, file_name))
        data = nii.get_data()

        mask = data[:, :, 0]
        mask = np.rot90(mask)
        plt.imshow(mask)
        plt.show()

def untar_spark_challenge_mask():
    in_mask_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal_mask_pair'
    in_dcm_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal'
    out_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal_mask'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_list = os.listdir(in_mask_dir)
    for file_name in file_list:
        if file_name.endswith('.tar'):
            tar_file = tarfile.open(os.path.join(in_mask_dir, file_name))
            tar_file.extractall(path=out_dir)
            tar_file.close()
            os.remove(os.path.join(out_dir, file_name.split('.')[0] + '.json'))
            mask_path = os.path.join(out_dir, file_name.split('.')[0] + '.nii.gz')
            mask_image = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_image)

            dcm_file_path = os.path.join(in_dcm_dir, file_name.split('_')[0] + '.dcm')
            # sitk_mr_file = sitk.ReadImage(dcm_file_path)
            # mr1 = sitk.GetArrayFromImage(sitk_mr_file)[0, :, :]

            pydicom_file = pydicom.read_file(dcm_file_path)
            mr = pydicom_file.pixel_array
            pixel_spacing = [float(pydicom_file.PixelSpacing[0]), float(pydicom_file.PixelSpacing[1])]
            mask_image.SetSpacing(pixel_spacing)
            sitk.WriteImage(mask_image, os.path.join(out_dir, file_name.split('_')[0] + '.nii.gz'))
            os.remove(mask_path)

def change_spark_mask():
    mask_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal_mask'
    file_list = os.listdir(mask_dir)
    for file_name in file_list:
        if file_name.endswith('.nii.gz'):
            nii = nib.load(os.path.join(mask_dir, file_name))
            mask = nii.get_data()
            mask = mask - 1
            mask = np.where(mask < 0, 0, mask)
            out_mask_nii = nib.Nifti1Image(mask.astype(np.int16), affine=nii.affine)
            nib.save(out_mask_nii, os.path.join(mask_dir, file_name))

if __name__ == '__main__':
    change_mask()
    # show_mask()



