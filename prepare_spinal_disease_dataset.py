import os
import numpy as np
from preprocessing.spinal_disease_dataset_processor import Spinal_disease_dataset_processor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str, default='DeepLabv3_plus_lil',
                        help="the model name, "
                             "DeepLabv3_plus"
                             "DeepLabv3_plus_gcn, "
                             "DeepLabv3_plus_lil")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/DGMSNet/Spinal_disease_dataset',
                        help="the data dir")

    args = parser.parse_args()

    data_root_dir = args.data_dir
    print('Prepare spinal disease dataset.')
    data_processor = Spinal_disease_dataset_processor(data_root_dir=data_root_dir)
    data_processor.merge_training_data()
    data_processor.extract_mid_sagittal_image_train201()
    data_processor.extract_mid_sagittal_image_testA50()
    data_processor.extract_mid_sagittal_image_testB50()
    data_processor.move_dataset()
    data_processor.make_h5_dataset()
    print('Done!')