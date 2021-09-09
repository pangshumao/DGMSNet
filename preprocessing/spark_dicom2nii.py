import SimpleITK as sitk
import pydicom
import os
import numpy as np

if __name__ == '__main__':
    dcm_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal'
    out_dir = 'L:\\ImageData\\Spine_Localization_PIL\\lumbar_testB50_mid_sagittal_nii'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dcm_file_list = os.listdir(dcm_dir)
    for dcm_file_name in dcm_file_list:
        if dcm_file_name.endswith('.dcm'):
            print(dcm_file_name)
            pydicom_file = pydicom.read_file(os.path.join(dcm_dir, dcm_file_name))
            itk_img = sitk.ReadImage(os.path.join(dcm_dir, dcm_file_name))
            mr = pydicom_file.pixel_array
            spacing = [float(pydicom_file.PixelSpacing[0]), float(pydicom_file.PixelSpacing[1])]

            out = sitk.GetImageFromArray(mr)
            out.SetSpacing(spacing)
            out.SetOrigin(itk_img.GetOrigin())
            sitk.WriteImage(out, os.path.join(out_dir, dcm_file_name.split('.')[0] + '.nii.gz'))




