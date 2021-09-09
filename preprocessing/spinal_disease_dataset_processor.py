import numpy as np
import os
import shutil
import json
from core.structure import construct_studies
from core.structure import construct_studies_for_testB
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from skimage import transform
from datasets.augment import resize_point
from datasets.utils import pad_image_coords_to_square
import h5py

def make_h5(data_dir, h5_group, case_inds, img_size, weak_label_num=0):
    weak_mr_dir = os.path.join(data_dir, 'in', 'weak_supervised_MR')
    mr_dir = os.path.join(data_dir, 'in', 'MR')
    mask_dir = os.path.join(data_dir, 'in', 'mask')
    keypoints_dir = os.path.join(data_dir, 'in', 'keypoints')

    sample_num = case_inds.size

    # For data
    flag = True
    for case_ind in case_inds:
        print('processing case%d' % case_ind)
        if case_ind > 416: # strongly-supervised dataset
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
            image = reader.Execute()
            mr = sitk.GetArrayFromImage(image)  # 1, h, w
            mr = mr[0, :, :]

            mask_image = sitk.ReadImage(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
            mask = sitk.GetArrayFromImage(mask_image)
            mask = pad_image_coords_to_square(mask)
            mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant',
                                    anti_aliasing=False).astype(np.int)
            mask = mask[None, :, :]
        else: # weakly-supervised dataset
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(weak_mr_dir, 'Case' + str(case_ind) + '.dcm'))
            image = reader.Execute()
            mr = sitk.GetArrayFromImage(image)  # 1, h, w
            mr = mr[0, :, :]
        kp_npz = np.load(os.path.join(keypoints_dir, 'Case' + str(case_ind) + '.npz'))
        coords = kp_npz['coords'] # (10, 2)
        mr, coords = pad_image_coords_to_square(mr, coords)
        # original_img_size = kp_npz['img_size'] # (2, ), [height, width]
        original_img_size = np.array(mr.shape)
        pixelspacing = kp_npz['pixelspacing']  # (2, ), [pixel_size_w, pixel_size_h]
        pixelspacing[0] = pixelspacing[0] * original_img_size[1] / img_size[1]
        pixelspacing[1] = pixelspacing[1] * original_img_size[0] / img_size[0]


        mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
        coords = resize_point(coords, in_img_size=original_img_size, out_img_size=img_size)

        # plt.subplot(121)
        # plt.imshow(mr, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(mask[0, :, :], cmap='gray')
        # plt.show()


        mr = mr[None, None, :, :]

        coords = coords[None, :, :] # (1, 10, 2)
        pixelspacing = pixelspacing[None, :] # (1, 2)
        original_img_size = original_img_size[None, :] # (1, 2)

        if flag:
            flag = False
            h5_group.create_dataset('mr', data=mr, maxshape=(sample_num, mr.shape[1], mr.shape[2], mr.shape[3]),
                                   chunks=(1, mr.shape[1], mr.shape[2], mr.shape[3]))
            h5_group.create_dataset('mask', data=mask, maxshape=(sample_num - weak_label_num, mask.shape[1], mask.shape[2]),
                                   chunks=(1, mask.shape[1], mask.shape[2]))
            h5_group.create_dataset('coords', data=coords, maxshape=(sample_num, coords.shape[1], coords.shape[2]),
                                    chunks=(1, coords.shape[1], coords.shape[2]))
            h5_group.create_dataset('pixelspacing', data=pixelspacing, maxshape=(sample_num, pixelspacing.shape[1]),
                                    chunks=(1, pixelspacing.shape[1]))
            h5_group.create_dataset('img_size', data=original_img_size, maxshape=(sample_num, original_img_size.shape[1]),
                                    chunks=(1, original_img_size.shape[1]))
        else:
            h5_group['mr'].resize(h5_group['mr'].shape[0] + mr.shape[0], axis=0)
            h5_group['mr'][-mr.shape[0]:] = mr

            if case_ind > 416:
                h5_group['mask'].resize(h5_group['mask'].shape[0] + mask.shape[0], axis=0)
                h5_group['mask'][-mask.shape[0]:] = mask

            h5_group['coords'].resize(h5_group['coords'].shape[0] + coords.shape[0], axis=0)
            h5_group['coords'][-coords.shape[0]:] = coords

            h5_group['pixelspacing'].resize(h5_group['pixelspacing'].shape[0] + pixelspacing.shape[0], axis=0)
            h5_group['pixelspacing'][-pixelspacing.shape[0]:] = pixelspacing

            h5_group['img_size'].resize(h5_group['img_size'].shape[0] + original_img_size.shape[0], axis=0)
            h5_group['img_size'][-original_img_size.shape[0]:] = original_img_size

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Spinal_disease_dataset_processor:
    def __init__(self, data_root_dir,
                 img_size=[512, 512],
                 train51_dir_name='lumbar_train51',
                 train150_dir_name='lumbar_train150',
                 train51_annotation_file_name='lumbar_train51_annotation.json',
                 train150_annotation_file_name='lumbar_train150_annotation.json',
                 testA50_dir_name='lumbar_testA50',
                 testB50_dir_name='lumbar_testB50',
                 testB50_series_map_name='testB50_series_map.json',
                 out_train201_dir_name='lumbar_train201',
                 out_train201_annotation_file_name='lumbar_train201_annotation.json',
                 out_train201_mid_sagittal_dir_name='lumbar_train201_mid_sagittal',
                 out_train201_keypoints_dir_name='lumbar_train201_keypoints',
                 out_testA50_mid_sagittal_dir_name='lumbar_testA50_mid_sagittal',
                 out_testB50_mid_sagittal_dir_name='lumbar_testB50_mid_sagittal'):
        self.img_size = img_size
        self.data_root_dir = data_root_dir
        self.train51_dir_name = train51_dir_name
        self.train150_dir_name = train150_dir_name
        self.train51_annotation_file_name = train51_annotation_file_name
        self.train150_annotation_file_name = train150_annotation_file_name
        self.testA50_dir_name = testA50_dir_name
        self.testB50_dir_name = testB50_dir_name
        self.testB50_series_map_name = testB50_series_map_name
        self.out_train201_dir_name = out_train201_dir_name
        self.out_train201_annotation_file_name = out_train201_annotation_file_name
        self.out_train201_mid_sagittal_dir_name = out_train201_mid_sagittal_dir_name
        self.out_train201_keypoints_dir_name = out_train201_keypoints_dir_name
        self.out_testA50_mid_sagittal_dir_name = out_testA50_mid_sagittal_dir_name
        self.out_testB50_mid_sagittal_dir_name = out_testB50_mid_sagittal_dir_name


    def merge_training_data(self):
        '''
        merge the train150 and train51 dataset to train201 dataset
        :return:
        '''
        print('Merging the train150 and train51 dataset to train201 dataset')

        root_dir = self.data_root_dir
        study_dir_1 = os.path.join(root_dir, self.train150_dir_name)
        study_dir_2 = os.path.join(root_dir, self.train51_dir_name)

        json_path_1 = os.path.join(root_dir, self.train150_annotation_file_name)
        json_path_2 = os.path.join(root_dir, self.train51_annotation_file_name)

        out_json_path = os.path.join(root_dir, self.out_train201_annotation_file_name)

        merge_train_dir = os.path.join(root_dir, self.out_train201_dir_name)

        print('The path of merged train201 dataset is : %s' % merge_train_dir)
        print('The path of merged train201 annotation is : %s' % out_json_path)
        if not os.path.exists(merge_train_dir):
            os.makedirs(merge_train_dir)
            for study in os.listdir(study_dir_1):
                shutil.copytree(os.path.join(study_dir_1, study), os.path.join(merge_train_dir, study))
            for study in os.listdir(study_dir_2):
                shutil.copytree(os.path.join(study_dir_2, study), os.path.join(merge_train_dir, study))

        with open(json_path_1, 'r') as file:
            annotations_1 = json.load(file)

        with open(json_path_2, 'r') as file:
            annotations_2 = json.load(file)

        merge_annotations = []

        for annotation in annotations_1:
            merge_annotations.append(annotation)

        for annotation in annotations_2:
            merge_annotations.append(annotation)

        with open(out_json_path, 'w') as file:
            json.dump(merge_annotations, file, cls=NpEncoder)

    def extract_mid_sagittal_image_train201(self):
        print('Extracting middle sagittal image for train201 dataset.')
        data_dir = self.data_root_dir
        out_dir = os.path.join(data_dir, self.out_train201_mid_sagittal_dir_name)
        kp_dir = os.path.join(data_dir, self.out_train201_keypoints_dir_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(kp_dir):
            os.makedirs(kp_dir)

        studies, annotations, counter = construct_studies(
            os.path.join(data_dir, self.out_train201_dir_name),
            os.path.join(data_dir, self.out_train201_annotation_file_name),
            multiprocessing=True)
        vertebrae_names = ['L1', 'L2', 'L3', 'L4', 'L5']
        disc_names = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

        case_ind = 216
        for key, annotation in annotations.items():
            print(case_ind)
            study_id, _, _ = key
            study = studies[study_id]
            middle_sagittal = study.t2_sagittal_middle_frame
            middle_sagittal_image = np.array(middle_sagittal.image)

            h, w = middle_sagittal_image.shape

            vertebrae_x = annotation[0][:, 0].numpy()
            vertebrae_y = annotation[0][:, 1].numpy()
            disc_x = annotation[1][1:, 0].numpy()
            disc_y = annotation[1][1:, 1].numpy()

            vertebrae_coords = np.stack([vertebrae_x, vertebrae_y], axis=1)
            disc_coords = np.stack([disc_x, disc_y], axis=1)
            coords = np.concatenate((vertebrae_coords, disc_coords), axis=0)

            # plt.imshow(middle_sagittal_image, cmap='gray')
            # for i in range(vertebrae_x.shape[0]):
            #     plt.scatter(x=vertebrae_x[i], y=vertebrae_y[i], c='r')
            #     plt.text(x=vertebrae_x[i] + 3, y=vertebrae_y[i] + 3, s=vertebrae_names[i])
            #
            # for i in range(disc_x.shape[0]):
            #     plt.scatter(x=disc_x[i], y=disc_y[i], c='r')
            #     plt.text(x=disc_x[i] + 3, y=disc_y[i] + 3, s=disc_names[i])
            # plt.show()

            pixel_spacing = list(middle_sagittal.pixel_spacing.numpy())
            # pixel_spacing[0] = pixel_spacing[0] * w / img_w
            # pixel_spacing[1] = pixel_spacing[1] * h / img_h
            file_path = middle_sagittal.file_path
            out_path = os.path.join(out_dir, 'Case' + str(case_ind) + '.dcm')
            shutil.copy(file_path, out_path)
            np.savez(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'), coords=coords, img_size=[h, w],
                     pixelspacing=pixel_spacing)
            case_ind += 1

    def extract_mid_sagittal_image_testA50(self):
        print('Extracting middle sagittal image for testA50 dataset.')
        out_dir = os.path.join(self.data_root_dir, self.out_testA50_mid_sagittal_dir_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        data = pd.read_excel(os.path.join(self.data_root_dir, 'testA50_sagittal_image_information.xlsx'))
        row_num = len(data)
        for i in range(row_num):
            study_name = data.loc[i][0]
            file_name = data.loc[i][1]
            out_file_name = 'Case' + str(data.loc[i][2]) + '.dcm'
            src_path = os.path.join(self.data_root_dir, self.testA50_dir_name, study_name, file_name)
            dst_path = os.path.join(out_dir, out_file_name)
            shutil.copy(src_path, dst_path)

    def extract_mid_sagittal_image_testB50(self):
        print('Extracting middle sagittal image for testB50 dataset.')
        out_dir = os.path.join(self.data_root_dir, self.out_testB50_mid_sagittal_dir_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        data = pd.read_excel(os.path.join(self.data_root_dir, 'testB50_sagittal_image_information.xlsx'))
        row_num = len(data)
        for i in range(row_num):
            study_name = data.loc[i][0]
            file_name = data.loc[i][1]
            out_file_name = 'Case' + str(data.loc[i][2]) + '.dcm'
            src_path = os.path.join(self.data_root_dir, self.testB50_dir_name, study_name, file_name)
            dst_path = os.path.join(out_dir, out_file_name)
            shutil.copy(src_path, dst_path)

    def mask2keypoints(self, mask_dir, kp_dir):
        '''
        extract the centroid for the mask.
        :param mask_dir:
        :param kp_dir:
        :return:
        '''
        mask_file_list = os.listdir(mask_dir)
        for mask_file_name in mask_file_list:
            if mask_file_name.endswith('nii.gz'):
                mask_image = sitk.ReadImage(os.path.join(mask_dir, mask_file_name))
                mask = sitk.GetArrayFromImage(mask_image) # (h, w)

                pixelspacing = mask_image.GetSpacing()
                height, width = mask.shape
                class_num = mask.max() + 1
                coords = np.zeros((class_num - 1, 2), dtype=np.float32)
                for i in range(1, class_num):
                    pos = np.argwhere(mask == i)
                    count = pos.shape[0]
                    center_h, center_w = pos.sum(0) / count
                    center_h = round(center_h)
                    center_w = round(center_w)
                    coords[i - 1, :] = np.array([center_w, center_h])  # x, y

                np.savez(os.path.join(kp_dir, mask_file_name.split('.')[0] + '.npz'), coords=coords, img_size=[height, width],
                         pixelspacing=[pixelspacing[0], pixelspacing[1]])

    def move_dataset(self):
        data_dir = self.data_root_dir
        train201_mr_dir = os.path.join(data_dir, self.out_train201_mid_sagittal_dir_name)
        kp_dir = os.path.join(data_dir, self.out_train201_keypoints_dir_name)
        testA50_mr_dir = os.path.join(data_dir, self.out_testA50_mid_sagittal_dir_name)
        testB50_mr_dir = os.path.join(data_dir, self.out_testB50_mid_sagittal_dir_name)

        out_dir = os.path.join(data_dir, 'in')

        out_weak_mr_dir = os.path.join(out_dir, 'weak_supervised_MR')
        out_kp_dir = os.path.join(out_dir, 'keypoints')
        out_mr_dir = os.path.join(out_dir, 'MR')

        shutil.copytree(train201_mr_dir, out_weak_mr_dir)
        # shutil.copytree(kp_dir, out_kp_dir)
        shutil.copytree(testA50_mr_dir, out_mr_dir)

        testB50_file_list = os.listdir(testB50_mr_dir)
        for file_name in testB50_file_list:
            shutil.copy(os.path.join(testB50_mr_dir, file_name), os.path.join(out_mr_dir, file_name))

    def make_h5_dataset(self):
        data_dir = self.data_root_dir

        img_size = self.img_size
        weak_label_num = 201

        split_data = np.load(os.path.join(data_dir, 'split_ind_fold1.npz'))
        train_ind = split_data['train_ind']
        val_ind = split_data['val_ind']
        test_ind = split_data['test_ind']

        weak_label_ind = np.array(range(216, 417))
        train_ind = np.concatenate((train_ind, weak_label_ind), axis=0)
        h5_dir = os.path.join(data_dir, 'in', 'h5py')
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)

        try:
            f = h5py.File(os.path.join(h5_dir, 'data_fold1.h5'), 'w')
            g_train = f.create_group('train')
            g_val = f.create_group('val')
            g_test = f.create_group('test')

            make_h5(data_dir, g_train, train_ind, img_size, weak_label_num)
            make_h5(data_dir, g_val, val_ind, img_size)
            make_h5(data_dir, g_test, test_ind, img_size)
        finally:
            f.close()





if __name__ == '__main__':
    data_root_dir = '/public/pangshumao/data/DGMSNet/Spinal_disease_dataset'
    data_processor = Spinal_disease_dataset_processor(data_root_dir=data_root_dir)
    # data_processor.merge_training_data()
    # data_processor.extract_mid_sagittal_image_train201()
    # data_processor.extract_mid_sagittal_image_testA50()
    # data_processor.extract_mid_sagittal_image_testB50()
    # data_processor.move_dataset()
    data_processor.make_h5_dataset()





