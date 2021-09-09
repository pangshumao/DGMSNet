import os
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, Union
from tqdm import tqdm
import torch
from torchvision.transforms import functional as tf
from .dicom import DICOM, lazy_property
from .series import Series
from ..data_utils import read_annotation, resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from core.data_utils import rotate_point
import json


class Study(dict):
    def __init__(self, study_dir, pool=None):
        dicom_list = []
        if pool is not None:
            async_results = []
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                async_results.append(pool.apply_async(DICOM, (dicom_path, )))

            for async_result in async_results:
                async_result.wait()
                dicom = async_result.get()
                dicom_list.append(dicom)
        else:
            for dicom_name in os.listdir(study_dir):
                dicom_path = os.path.join(study_dir, dicom_name)
                dicom = DICOM(dicom_path)
                dicom_list.append(dicom)

        dicom_dict = {}
        for dicom in dicom_list:
            series_uid = dicom.series_uid
            if series_uid not in dicom_dict:
                dicom_dict[series_uid] = [dicom]
            else:
                dicom_dict[series_uid].append(dicom)

        super().__init__({k: Series(v) for k, v in dicom_dict.items()})

        self.t2_sagittal_uid = None
        self.t2_transverse_uid = None
        # 通过平均值最大的来剔除压脂项
        max_t2_sagittal_mean = 0
        max_t2_transverse_mean = 0

        for series_uid, series in self.items():
            if series.plane == 'sagittal' and series.t_type == 'T2':
                t2_sagittal_mean = series.mean
                if t2_sagittal_mean > max_t2_sagittal_mean:
                    max_t2_sagittal_mean = t2_sagittal_mean
                    self.t2_sagittal_uid = series_uid
            if series.plane == 'transverse' and series.t_type == 'T2':
                t2_transverse_mean = series.mean
                if t2_transverse_mean > max_t2_transverse_mean:
                    max_t2_transverse_mean = t2_transverse_mean
                    self.t2_transverse_uid = series_uid

        if self.t2_sagittal_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'sagittal':
                    t2_sagittal_mean = series.mean
                    if t2_sagittal_mean > max_t2_sagittal_mean:
                        max_t2_sagittal_mean = t2_sagittal_mean
                        self.t2_sagittal_uid = series_uid

        if self.t2_transverse_uid is None:
            for series_uid, series in self.items():
                if series.plane == 'transverse':
                    t2_transverse_mean = series.mean
                    if t2_transverse_mean > max_t2_transverse_mean:
                        max_t2_transverse_mean = t2_transverse_mean
                        self.t2_transverse_uid = series_uid

    @lazy_property
    def study_uid(self):
        study_uid_counter = Counter([s.study_uid for s in self.values()])
        return study_uid_counter.most_common(1)[0][0]

    @property
    def t2_sagittal(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_sagittal_uid is None:
            return None
        else:
            return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_transverse_uid is None:
            return None
        else:
            return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> Union[None, DICOM]:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame

    def set_t2_sagittal_middle_frame(self, series_uid, instance_uid):
        assert series_uid in self
        self.t2_sagittal_uid = series_uid
        self.t2_sagittal.set_middle_frame(instance_uid)

    def set_t2_sagittal(self, series_uid):
        assert series_uid in self
        self.t2_sagittal_uid = series_uid

    def t2_transverse_k_nearest(self, pixel_coord, k, size, max_dist, coord_ind, random_state, prob_rotate=0,
                                max_angel=0):
        """

        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param max_dist:
        :param coord_ind:
        :param prob_rotate:
        :param max_angel:
        :return: 图像张量(k, size[0], size[1])
        """
        # height = 345
        # width = 345
        pixel_h = 0.625
        pixel_w = 0.625
        if self.t2_transverse is None:
            # padding
            images = torch.zeros(k, *size, dtype=torch.uint8)
            return images
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        dicoms = self.t2_transverse.k_nearest(human_coord, k, max_dist)

        point = human_coord[coord_ind]
        series = dicoms[coord_ind]

        images = []
        inds = []
        count = 0
        flag = True
        for dicom in series:
            if dicom is None:
                image = np.zeros((512, 512), dtype=np.uint8)
                inds.append(count)
            else:
                image = dicom.image

                # crop_size = min(h, w)
                # image = tf.crop(image, 0, 0, crop_size, crop_size)
                # plt.imshow(np.array(image), cmap='gray')
                # plt.show()
                if flag:
                    flag = False
                    pixel_spacing = dicom.pixel_spacing.numpy()
                    w, h = image.size
                    height = int(round(h * pixel_spacing[1] / pixel_h))
                    width = int(round(w * pixel_spacing[0] / pixel_w))
                    projection = dicom.projection(point)
                    w, h = image.size
                    height_ratio = height / h
                    width_ratio = width / w
                    ratio = torch.tensor([width_ratio, height_ratio])
                    projection = (projection * ratio).round().long()
                image = tf.resize(image, size=(height, width))

                image = np.array(image)
                # plt.imshow(image, cmap='gray')
                # plt.show()
            images.append(image)
            count += 1

        if len(inds) > 0 and flag is False:
            for ind in inds:
                images[ind] = np.resize(images[ind], (height, width))
        images = np.stack(images, axis=0)
        if flag:
            crop = images[:, images.shape[1] // 2 - size[0] // 2: images.shape[1] // 2 - size[0] // 2 + size[0],
                   images.shape[2] // 2 - size[1] // 2: images.shape[2] // 2 - size[1] // 2 + size[1]]

            # plt.imshow(crop[1], cmap='gray')
            # plt.show()

            crop = torch.from_numpy(crop)
            return crop
        else:
            if max_angel > 0 and random_state.uniform() <= prob_rotate:
                center = torch.tensor(images.shape[1:], dtype=torch.float32) / 2
                angel = random_state.randint(-max_angel, max_angel)
                projection = rotate_point(projection, angel, center).long()
                images = rotate(images, angle=angel, axes=(2, 1), reshape=False, order=3, mode='reflect', cval=-1)

            if projection[0].item() == 0 and projection[1].item() == 0:
                return torch.zeros(k, *size, dtype=torch.uint8)
            start_h = max(0, projection[1].item() - size[0] // 2)
            start_w = max(0, projection[0].item() - size[1] // 2)
            crop = images[:, start_h: start_h + size[0],
                   start_w: start_w + size[1]]

            # plt.imshow(crop[1], cmap='gray')
            # plt.show()
            crop = torch.from_numpy(crop)
            d, h, w = crop.shape
            out = torch.zeros(k, *size, dtype=crop.dtype)
            out[:, (size[0] - h) // 2: (size[0] - h) // 2 + h, (size[1] - w) // 2: (size[1] - w) // 2 + w] = crop

            # plt.imshow(out[1].numpy(), cmap='gray')
            # plt.show()
            return out

    def t2_transverse_k_nearest_all_coord(self, pixel_coord, k, size, max_dist, random_state, prob_rotate=0,
                                max_angel=0, n_holes=0):
        '''
        crop the transverse image for all coord
        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param max_dist:
        :param random_state:
        :param prob_rotate:
        :param max_angel:
        :return: 图像张量(M, k, size[0], size[1])
        '''
        # height = 345
        # width = 345
        pixel_h = 0.625
        pixel_w = 0.625
        if self.t2_transverse is None:
            # padding
            images = torch.zeros(pixel_coord.shape[0], k, *size, dtype=torch.uint8)
            return images
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)

        # plt.imshow(np.array(self.t2_sagittal_middle_frame.image), cmap='gray')
        # plt.show()
        #
        # transverse = self.t2_transverse
        # for dicom in transverse:
        #     image = np.array(dicom.image)
        #     plt.imshow(image, cmap='gray')
        #     plt.show()

        dicoms = self.t2_transverse.k_nearest(human_coord, k, max_dist)

        all_images = []
        for point, series in zip(human_coord, dicoms):
            images = []
            inds = []
            count = 0
            flag = True
            for dicom in series:
                if dicom is None:
                    inds.append(count)
                    image = np.zeros((512, 512), dtype=np.uint8)
                else:
                    image = dicom.image

                    # crop_size = min(h, w)
                    # image = tf.crop(image, 0, 0, crop_size, crop_size)
                    # plt.imshow(np.array(image), cmap='gray')
                    # plt.show()
                    if flag:
                        flag = False
                        pixel_spacing = dicom.pixel_spacing.numpy()
                        w, h = image.size
                        height = int(round(h * pixel_spacing[1] / pixel_h))
                        width = int(round(w * pixel_spacing[0] / pixel_w))

                        # print('transverse image height = %d, width = %d' % (height, width))

                        projection = dicom.projection(point)
                        w, h = image.size
                        height_ratio = height / h
                        width_ratio = width / w
                        ratio = torch.tensor([width_ratio, height_ratio])
                        projection = (projection * ratio).round().long()
                    image = tf.resize(image, size=(height, width))

                    image = np.array(image)
                    # plt.imshow(image, cmap='gray')
                    # plt.show()
                images.append(image)
                count += 1
            if len(inds) > 0 and flag is False:
                for ind in inds:
                    images[ind] = np.resize(images[ind], (height, width))
            images = np.stack(images, axis=0)
            if flag:
                crop = images[:, images.shape[1] // 2 - size[0] // 2: images.shape[1] // 2 - size[0] // 2 + size[0],
                       images.shape[2] // 2 - size[1] // 2: images.shape[2] // 2 - size[1] // 2 + size[1]]

                # plt.imshow(crop[1], cmap='gray')
                # plt.show()

                crop = torch.from_numpy(crop)
                all_images.append(crop)
            else:
                if max_angel > 0 and random_state.uniform() <= prob_rotate:
                    center = torch.tensor(images.shape[1:], dtype=torch.float32) / 2
                    angel = random_state.randint(-max_angel, max_angel)
                    projection = rotate_point(projection, angel, center).long()
                    images = rotate(images, angle=angel, axes=(2, 1), reshape=False, order=3, mode='reflect', cval=-1)

                if projection[0].item() == 0 and projection[1].item() == 0:
                    all_images.append(torch.zeros(k, *size, dtype=torch.uint8))
                else:
                    start_h = max(0, projection[1].item() - size[0] // 2)
                    start_w = max(0, projection[0].item() - size[1] // 2)
                    crop = images[:, start_h: start_h + size[0],
                           start_w: start_w + size[1]]

                    # plt.imshow(crop[1], cmap='gray')
                    # plt.show()
                    crop = torch.from_numpy(crop)
                    d, h, w = crop.shape
                    out = torch.zeros(k, *size, dtype=crop.dtype)
                    out[:, (size[0] - h) // 2: (size[0] - h) // 2 + h, (size[1] - w) // 2: (size[1] - w) // 2 + w] = crop

                    # plt.imshow(out[1].numpy(), cmap='gray')
                    # plt.show()
                    all_images.append(out)
        all_images = torch.stack(all_images, dim=0)
        if n_holes > 0 and random_state.uniform() <= prob_rotate:
            all_images = self.cutout(all_images, n_holes, window_size=10, random_state=random_state)
        return all_images

    def t2_sagittal_k_nearest(self, pixel_coord, k, size,  random_state, prob_rotate=0,
                                max_angel=0) -> (torch.Tensor, torch.Tensor):
        """

        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param max_dist:
        :param prob_rotate:
        :param max_angel:
        :return: 图像张量(1, k, size[0], size[1])
        """
        # height = 512
        # width = 512
        pixel_h = 0.6875
        pixel_w = 0.6875

        t2_sagittal = self.t2_sagittal
        slice_num = len(t2_sagittal)
        start_slice = (slice_num - k) // 2
        end_slice = start_slice + k
        slices = []
        flag = True
        for i in range(start_slice, end_slice):
            dicom = t2_sagittal[i]
            slice = dicom.image

            if flag:
                flag = False
                w, h = slice.size
                pixel_spacing = dicom.pixel_spacing.numpy()
                height = int(round(h * pixel_spacing[1] / pixel_h))
                width = int(round(w * pixel_spacing[0] / pixel_w))
                height_ratio = height / h
                width_ratio = width / w
                ratio = torch.tensor([width_ratio, height_ratio])
                pixel_coord = (pixel_coord * ratio).round().long()

            slice = tf.resize(slice, size=(height, width))
            slice = np.array(slice)

            slices.append(slice)
        slices = np.stack(slices, axis=0)

        # plt.imshow(slices[3], cmap='gray')
        # plt.scatter(x=pixel_coord[:, 0], y=pixel_coord[:, 1])
        # plt.show()

        if max_angel > 0 and random_state.uniform() <= prob_rotate:
            center = torch.tensor(slices.shape[1:], dtype=torch.float32) / 2
            angel = random_state.randint(-max_angel, max_angel)
            pixel_coord = rotate_point(pixel_coord, angel, center).long()
            slices = rotate(slices, angle=angel, axes=(2, 1), reshape=False, order=3, mode='reflect', cval=-1)
        coord_ind = random_state.randint(0, pixel_coord.shape[0])

        if pixel_coord[coord_ind, 0].item() == 0 and pixel_coord[coord_ind, 1].item() == 0:
            return torch.zeros(1, k, *size, dtype=torch.uint8), coord_ind

        start_h = max(0, pixel_coord[coord_ind, 1].item() - size[0] // 2)
        start_w = max(0, pixel_coord[coord_ind, 0].item() - size[1] // 2)
        crop = slices[:, start_h : start_h + size[0],
               start_w : start_w + size[1]]

        # plt.imshow(crop[4], cmap='gray')
        # plt.show()

        crop = torch.from_numpy(crop)

        d, h, w = crop.shape
        out = torch.zeros(k, *size, dtype=crop.dtype)
        out[:, (size[0] - h) // 2: (size[0] - h) // 2 + h, (size[1] - w) // 2: (size[1] - w) // 2 + w] = crop

        out = out.unsqueeze(0)

        return out, coord_ind

    def t2_sagittal_k_nearest_all_coord(self, pixel_coord, k, size,  random_state, prob_rotate=0, max_angel=0, n_holes=0,
                                        return_mask=False, mask_radius=6, labels=None):
        """
        crop sagittal image for all coord
        :param pixel_coord: (M, 2)
        :param k:
        :param size:
        :param max_dist:
        :param prob_rotate:
        :param max_angel:
        :return: 图像张量(M, 1, k, size[0], size[1])
        """

        # height = 512
        # width = 512
        pixel_h = 0.6875
        pixel_w = 0.6875
        t2_sagittal = self.t2_sagittal
        slice_num = len(t2_sagittal)
        start_slice = (slice_num - k) // 2
        end_slice = start_slice + k
        slices = []
        flag = True
        for i in range(start_slice, end_slice):
            dicom = t2_sagittal[i]
            slice = dicom.image

            if flag:
                flag = False
                pixel_spacing = dicom.pixel_spacing.numpy()
                w, h = slice.size
                height = int(round(h * pixel_spacing[1] / pixel_h))
                width = int(round(w * pixel_spacing[0] / pixel_w))

                # print('sagittal image height = %d, width = %d' % (height, width))

                height_ratio = height / h
                width_ratio = width / w
                ratio = torch.tensor([width_ratio, height_ratio])
                pixel_coord = (pixel_coord * ratio).round().long()

            slice = tf.resize(slice, size=(height, width))
            slice = np.array(slice)

            slices.append(slice)
        slices = np.stack(slices, axis=0)

        # plt.imshow(slices[3], cmap='gray')
        # plt.scatter(x=pixel_coord[:, 0], y=pixel_coord[:, 1])
        # plt.show()

        if max_angel > 0 and random_state.uniform() <= prob_rotate:
            center = torch.tensor(slices.shape[1:], dtype=torch.float32) / 2
            angel = random_state.randint(-max_angel, max_angel)
            pixel_coord = rotate_point(pixel_coord, angel, center).long()
            slices = rotate(slices, angle=angel, axes=(2, 1), reshape=False, order=3, mode='reflect', cval=-1)

        all_images = []
        all_masks = []
        for coord_ind in range(pixel_coord.shape[0]):
            if pixel_coord[coord_ind, 0].item() == 0 and pixel_coord[coord_ind, 1].item() == 0:
                all_images.append(torch.zeros(1, k, *size, dtype=torch.uint8))
                all_masks.append(torch.zeros(k, *size, dtype=torch.uint8))
            else:
                start_h = max(0, pixel_coord[coord_ind, 1].item() - size[0] // 2)
                start_w = max(0, pixel_coord[coord_ind, 0].item() - size[1] // 2)
                crop = slices[:, start_h : start_h + size[0],
                       start_w : start_w + size[1]]

                # plt.subplot(121)
                # plt.imshow(crop[4], cmap='gray')
                # plt.show()

                crop = torch.from_numpy(crop)

                d, h, w = crop.shape
                out = torch.zeros(k, *size, dtype=crop.dtype)
                out[:, (size[0] - h) // 2: (size[0] - h) // 2 + h, (size[1] - w) // 2: (size[1] - w) // 2 + w] = crop
                out = out.unsqueeze(0)
                all_images.append(out)

                if return_mask:
                    mask = torch.zeros(k, *size, dtype=torch.uint8)
                    center_h = size[0] // 2
                    center_w = size[1] // 2
                    if isinstance(mask_radius, list) or isinstance(mask_radius, tuple):
                        mask[:, center_h - mask_radius[labels[coord_ind]]:
                                center_h + mask_radius[labels[coord_ind]],
                                center_w - mask_radius[labels[coord_ind]]:
                                center_w + mask_radius[labels[coord_ind]]] = labels[coord_ind].item() + 1
                    else:
                        mask[:, center_h - mask_radius:
                                center_h + mask_radius,
                                center_w - mask_radius:
                                center_w + mask_radius] = labels[coord_ind].item() + 1

                    # plt.subplot(122)
                    # plt.imshow(mask[4], cmap='gray')
                    # plt.show()

                    all_masks.append(mask)
        all_images = torch.stack(all_images, dim=0)
        if return_mask:
            all_masks = torch.stack(all_masks, dim=0)
        if n_holes > 0 and random_state.uniform() <= prob_rotate:
            all_images = self.cutout(all_images, n_holes, window_size=10, random_state=random_state)
        if return_mask:
            return all_images, all_masks
        else:
            return all_images

    def cutout(self, img, n_holes, window_size, random_state):
        '''

        :param img: Tensor with shape of (n, c, d, h, w) or (n, c, h, w)
        :param n_holes: int
        :param window_size: int
        :param random_state:
        :return:
        '''
        assert img.dim() in [4, 5]
        if img.dim() == 5:
            n, c, d, h, w = img.shape
            mask = np.ones((d, h, w), np.float32)

            for i in range(n_holes):
                z = random_state.randint(d)
                y = random_state.randint(h)
                x = random_state.randint(w)

                z1 = np.clip(z - window_size // 2, 0, d)
                z2 = np.clip(z + window_size // 2, 0, d)
                y1 = np.clip(y - window_size // 2, 0, h)
                y2 = np.clip(y + window_size // 2, 0, h)
                x1 = np.clip(x - window_size // 2, 0, w)
                x2 = np.clip(x + window_size // 2, 0, w)

                mask[z1 : z2, y1 : y2, x1 : x2] = 0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        elif img.dim() == 4:
            n, c, h, w = img.shape
            mask = np.ones((h, w), np.float32)

            for i in range(n_holes):
                y = random_state.randint(h)
                x = random_state.randint(w)

                y1 = np.clip(y - window_size // 2, 0, h)
                y2 = np.clip(y + window_size // 2, 0, h)
                x1 = np.clip(x - window_size // 2, 0, w)
                x2 = np.clip(x + window_size // 2, 0, w)

                mask[y1: y2, x1: x2] = 0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        img = img.type(dtype=torch.uint8)
        return img


def _construct_studies(data_dir, multiprocessing=False):
    studies: Dict[str, Study] = {}
    if multiprocessing:
        pool = Pool(cpu_count())
    else:
        pool = None

    for study_name in tqdm(os.listdir(data_dir), ascii=True):
        study_dir = os.path.join(data_dir, study_name)
        study = Study(study_dir, pool)
        studies[study.study_uid] = study

    if pool is not None:
        pool.close()
        pool.join()
    return studies


def _set_middle_frame(studies: Dict[str, Study], annotation):
    counter = {
        't2_sagittal_not_found': [],
        't2_sagittal_miss_match': [],
        't2_sagittal_middle_frame_miss_match': []
    }
    for k in annotation.keys():
        if k[0] in studies:
            study = studies[k[0]]
            if study.t2_sagittal is None:
                counter['t2_sagittal_not_found'].append(study.study_uid)
            elif study.t2_sagittal_uid != k[1]:
                # print(study.t2_sagittal_uid)
                counter['t2_sagittal_miss_match'].append(study.study_uid)
            else:
                t2_sagittal = study.t2_sagittal
                gt_z_index = t2_sagittal.instance_uids[k[2]]
                middle_frame = t2_sagittal.middle_frame
                z_index = t2_sagittal.instance_uids[middle_frame.instance_uid]
                if abs(gt_z_index - z_index) > 1:
                    counter['t2_sagittal_middle_frame_miss_match'].append(study.study_uid)
            study.set_t2_sagittal_middle_frame(k[1], k[2])
    return counter


def construct_studies(data_dir, annotation_path=None, multiprocessing=False):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param multiprocessing:
    :param annotation_path: 如果有标注，那么根据标注来确定定位帧
    :return:
    """
    studies = _construct_studies(data_dir, multiprocessing)

    # 使用annotation制定正确的中间帧
    if annotation_path is None:
        return studies
    else:
        annotation = read_annotation(annotation_path)
        counter = _set_middle_frame(studies, annotation)
        return studies, annotation, counter

def construct_studies_for_testB(data_dir, series_map_path=None, multiprocessing=False):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param multiprocessing:
    :param series_map_path: for sagittal series
    :return:
    """
    studies = _construct_studies(data_dir, multiprocessing)

    if series_map_path is None:
        return studies
    else:
        with open(series_map_path, 'r') as file:
            series_maps = json.load(file)
            for series in series_maps:
                studyUid = series['studyUid']
                seriesUid = series['seriesUid']
                studies[studyUid].set_t2_sagittal(seriesUid)

        return studies
