import os
import numpy as np
from preprocessing.spinal_disease_dataset_processor import Spinal_disease_dataset_processor

if __name__ == '__main__':
    data_root_dir = '/public/pangshumao/data/DGMSNet/Spinal_disease_dataset'
    print('Prepare spinal disease dataset.')
    data_processor = Spinal_disease_dataset_processor(data_root_dir=data_root_dir)
    data_processor.merge_training_data()
    data_processor.extract_mid_sagittal_image_train201()
    data_processor.extract_mid_sagittal_image_testA50()
    data_processor.extract_mid_sagittal_image_testB50()
    data_processor.move_dataset()
    data_processor.make_h5_dataset()
    print('Done!')