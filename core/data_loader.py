from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from core.data_utils import gen_mask
from core.structure import DICOM, Study
from collections import OrderedDict
import os
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TKAgg')

class KeyPointDataSet(Dataset):
    def __init__(self,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 sagittal_size: Tuple[int, int]):
        '''
        Dataset for localizing key points of vertebrae and discs
        :param studies:
        :param annotations:
        :param prob_rotate:
        :param max_angel:
        :param num_rep:
        :param sagittal_size:
        '''
        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

        self.prob_rotate = prob_rotate
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.sagittal_size = sagittal_size
        self.random_state = np.random.RandomState(seed=10)

    def __len__(self):
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item) -> (Study, Any, (torch.Tensor, torch.Tensor)):
        item = item % len(self.annotations)
        key, (v_annotation, d_annotation) = self.annotations[item]
        return self.studies[key[0]], key, v_annotation, d_annotation

    def collate_fn(self, data) -> (Tuple[torch.Tensor], Tuple[None]):
        sagittal_images, vertebra_labels, disc_labels, distmaps = [], [], [], []
        for study, key, v_anno, d_anno in data:

            dicom: DICOM = study[key[1]][key[2]]
            pixel_coord = torch.cat([v_anno[:, :2], d_anno[:, :2]], dim=0)
            sagittal_image, pixel_coord, distmap = dicom.transform(
                pixel_coord, self.random_state, self.sagittal_size, self.prob_rotate, self.max_angel, distmap=True)

            # s_img = sagittal_image.to('cpu').numpy()[0]
            # d_map = distmap.to('cpu').numpy()
            # out_map = np.zeros(d_map.shape[1:])
            # for i in range(d_map.shape[0]):
            #     out_map += d_map[i]
            #
            # plt.subplot(121)
            # plt.imshow(s_img)
            # for i in range(0, 11):
            #     plt.scatter(x=pixel_coord[i, 0], y=pixel_coord[i, 1], c='r')
            #     # plt.text(x=pixel_coord[i, 0] + 2, y=pixel_coord[i, 1] + 2, s=names[i], c='w')
            #
            # plt.subplot(122)
            # plt.imshow(out_map)
            # plt.show()

            sagittal_images.append(sagittal_image)
            distmaps.append(distmap)

            v_label = torch.cat([pixel_coord[:v_anno.shape[0]], v_anno[:, 2:]], dim=-1)
            d_label = torch.cat([pixel_coord[v_anno.shape[0]:], d_anno[:, 2:]], dim=-1)
            vertebra_labels.append(v_label)
            disc_labels.append(d_label)

        sagittal_images = torch.stack(sagittal_images, dim=0) # (batch, 1, h, w)
        distmaps = torch.stack(distmaps, dim=0) # (batch, class_num, h, w)
        vertebra_labels = torch.stack(vertebra_labels, dim=0) # (batch, 5, k_nearest)
        disc_labels = torch.stack(disc_labels, dim=0) # (batch, 6, k_nearest)


        data = OrderedDict()
        data['sagittal_images'] = sagittal_images # (batch, 1, h, w)
        data['heatmaps'] = distmaps  # (batch, class_num, h, w)
        data['vertebra_labels'] = vertebra_labels # (batch, 5, k_nearest)
        data['disc_labels'] = disc_labels # (batch, 6, k_nearest)
        return data


class KeyPointDataLoader(DataLoader):
    # TODO 添加一些sampling的方法
    def __init__(self, studies, annotations, batch_size, sagittal_size,
                 num_workers=0, prob_rotate=0.0, max_angel=0, num_rep=1, pin_memory=False):
        dataset = KeyPointDataSet(studies=studies, annotations=annotations, sagittal_size=sagittal_size,
                             prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=dataset.collate_fn)


class DisDataSet(Dataset):
    def __init__(self,
                 studies,
                 annotations,
                 prob_rotate,
                 max_angel,
                 num_rep,
                 sagittal_size,
                 transverse_size,
                 v_k_nearest,
                 d_k_nearest,
                 max_dist,
                 classifier,
                 n_holes,
                 return_mask,
                 mask_radius):
        '''
        Dataset for disease prediction
        :param studies:
        :param annotations:
        :param prob_rotate:
        :param max_angel:
        :param num_rep:
        :param sagittal_size:
        :param transverse_size:
        :param v_k_nearest:
        :param d_k_nearest:
        :param max_dist:
        :param classifier: 'disc', 'ver'
        '''

        self.studies = studies
        self.annotations = []
        for k, annotation in annotations.items():
            study_uid, series_uid, instance_uid = k
            if study_uid not in self.studies:
                continue
            study = self.studies[study_uid]
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                self.annotations.append((k, annotation))

        self.prob_rotate = prob_rotate
        self.max_angel = max_angel
        self.num_rep = num_rep
        self.sagittal_size = sagittal_size
        self.transverse_size = transverse_size
        self.v_k_nearest = v_k_nearest
        self.d_k_nearest = d_k_nearest
        self.max_dist = max_dist
        self.random_state = np.random.RandomState(seed=10)
        self.classifier = classifier
        self.n_holes = n_holes
        self.return_mask = return_mask
        self.mask_radius = mask_radius

    def __len__(self):
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item) -> (Study, Any, (torch.Tensor, torch.Tensor)):
        item = item % len(self.annotations)
        key, (v_annotation, d_annotation) = self.annotations[item]
        return self.studies[key[0]], key, v_annotation, d_annotation

    def collate_fn(self, data):
        sagittal_images, transverse_images, vertebra_labels, disc_labels, schmor_labels = [], [], [], [], []
        if self.return_mask:
            masks = []

        if self.classifier == 'ver':
            for study, key, v_anno, d_anno in data:
                if self.return_mask:
                    sagittal_image, mask = study.t2_sagittal_k_nearest_all_coord(v_anno[:, :2], k=self.v_k_nearest,
                                                                           size=self.sagittal_size,
                                                                            random_state=self.random_state,
                                                                            prob_rotate=self.prob_rotate,
                                                                           max_angel=self.max_angel, n_holes=self.n_holes,
                                                                           labels=v_anno[:, 2],
                                                                           return_mask=self.return_mask,
                                                                                 mask_radius=self.mask_radius)
                    masks.append(mask)
                else:
                    sagittal_image = study.t2_sagittal_k_nearest_all_coord(v_anno[:, :2], k=self.v_k_nearest,
                                                                           size=self.sagittal_size,
                                                                           random_state=self.random_state,
                                                                           prob_rotate=self.prob_rotate,
                                                                           max_angel=self.max_angel,
                                                                           n_holes=self.n_holes)

                sagittal_images.append(sagittal_image)
                vertebra_labels.append(v_anno[:, 2])

            data = OrderedDict()
            sagittal_images = torch.cat(sagittal_images,
                                          dim=0)  # (batch * 5, 1, v_k_nearest, sagittal_size[0], sagittal_size[1])
            vertebra_labels = torch.cat(vertebra_labels, dim=0)  # (batch * 5, )
            if self.return_mask:
                masks = torch.cat(masks, dim=0)
                data['masks'] = masks # (batch, v_k_nearest, sagittal_size[0], sagittal_size[1])


            data['sagittal_images'] = sagittal_images  # (batch, 1, v_k_nearest, sagittal_size[0], sagittal_size[1])
            data['vertebra_labels'] = vertebra_labels  # (batch, )
        elif self.classifier == 'disc':
            for study, key, v_anno, d_anno in data:
                if self.return_mask:
                    sagittal_image, mask = study.t2_sagittal_k_nearest_all_coord(d_anno[:, :2], k=self.v_k_nearest,
                                                                           size=self.sagittal_size,
                                                                            random_state=self.random_state,
                                                                            prob_rotate=self.prob_rotate,
                                                                           max_angel=self.max_angel, n_holes=self.n_holes,
                                                                           labels=d_anno[:, 2],
                                                                           return_mask=self.return_mask,
                                                                                 mask_radius=self.mask_radius)
                    masks.append(mask)
                else:
                    sagittal_image = study.t2_sagittal_k_nearest_all_coord(d_anno[:, :2], k=self.v_k_nearest,
                                                                           size=self.sagittal_size,
                                                                            random_state=self.random_state,
                                                                            prob_rotate=self.prob_rotate,
                                                                           max_angel=self.max_angel, n_holes=self.n_holes)
                    transverse_image = study.t2_transverse_k_nearest_all_coord(
                        d_anno[:, :2], k=self.d_k_nearest, size=self.transverse_size, max_dist=self.max_dist,
                        random_state=self.random_state,
                        prob_rotate=self.prob_rotate, max_angel=self.max_angel, n_holes=self.n_holes
                    )

                    transverse_images.append(transverse_image)

                sagittal_images.append(sagittal_image)


                disc_labels.append(d_anno[:,  2])
                schmor_labels.append(d_anno[:, 3])

            sagittal_images = torch.cat(sagittal_images,
                                          dim=0)  # (batch * 6, 1, d_k_nearest, sagittal_size[0], sagittal_size[1])

            disc_labels = torch.cat(disc_labels, dim=0)  # (batch * 6, )
            schmor_labels = torch.cat(schmor_labels, dim=0)  # (batch * 6, )

            data = OrderedDict()
            if self.return_mask:
                masks = torch.cat(masks, dim=0)
                data['masks'] = masks # (batch, v_k_nearest, sagittal_size[0], sagittal_size[1])
            else:
                transverse_images = torch.cat(transverse_images,
                                              dim=0)  # (batch * 6, d_k_nearest, 1, transverse_size[0], transverse_size[1])
                data['transverse_images'] = transverse_images  # (batch, d_k_nearest, transverse_size[0], transverse_size[1])
            data['sagittal_images'] = sagittal_images # (batch, 1, d_k_nearest, sagittal_size[0], sagittal_size[1])

            data['disc_labels'] = disc_labels # (batch, )
            data['schmor_labels'] = schmor_labels
        elif self.classifier == 'schmor':
            for study, key, v_anno, d_anno in data:
                sagittal_image, mask = study.t2_sagittal_k_nearest_all_coord(d_anno[:, :2], k=self.v_k_nearest,
                                                                             size=self.sagittal_size,
                                                                             random_state=self.random_state,
                                                                             prob_rotate=self.prob_rotate,
                                                                             max_angel=self.max_angel,
                                                                             n_holes=self.n_holes,
                                                                             labels=d_anno[:, 3],
                                                                             return_mask=self.return_mask,
                                                                             mask_radius=self.mask_radius)
                masks.append(mask)
                sagittal_images.append(sagittal_image)

                disc_labels.append(d_anno[:, 2])
                schmor_labels.append(d_anno[:, 3])

            sagittal_images = torch.cat(sagittal_images,
                                        dim=0)  # (batch * 6, 1, d_k_nearest, sagittal_size[0], sagittal_size[1])

            disc_labels = torch.cat(disc_labels, dim=0)  # (batch * 6, )
            schmor_labels = torch.cat(schmor_labels, dim=0)  # (batch * 6, )

            data = OrderedDict()

            masks = torch.cat(masks, dim=0)
            data['masks'] = masks  # (batch, v_k_nearest, sagittal_size[0], sagittal_size[1])

            data['sagittal_images'] = sagittal_images  # (batch, 1, d_k_nearest, sagittal_size[0], sagittal_size[1])

            data['disc_labels'] = disc_labels  # (batch, )
            data['schmor_labels'] = schmor_labels
        return data


class DisDataLoader(DataLoader):
    # TODO 添加一些sampling的方法
    def __init__(self, studies, annotations, batch_size, sagittal_size, transverse_size, v_k_nearest, d_k_nearest,
                 classifier,
                 num_workers=0, prob_rotate=0.0, max_angel=0, n_holes=0, max_dist=8, num_rep=1, pin_memory=False,
                 return_mask=False, mask_radius=6):
        dataset = DisDataSet(studies=studies, annotations=annotations,
                             sagittal_size=sagittal_size,
                             transverse_size=transverse_size, v_k_nearest=v_k_nearest,
                             d_k_nearest=d_k_nearest, prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep, max_dist=max_dist, classifier=classifier,
                             n_holes=n_holes, return_mask=return_mask, mask_radius=mask_radius)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=dataset.collate_fn)

if __name__ == '__main__':
    from core.structure import construct_studies
    from core.data_utils import split_val_test, split_val_test_json

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    torch.manual_seed(seed=0)

    data_dir = '/public/pangshumao/data/Spark_Challenge'


    train51_studies, train51_annotation, train51_counter = construct_studies(
        os.path.join(data_dir, 'in', 'lumbar_train51'), os.path.join(data_dir, 'in', 'lumbar_train51_annotation.json'),
        multiprocessing=True)

    # val_studies, val_annotations, test_studies, test_annotations = split_val_test(train51_studies, train51_annotation, val_ratio=0.5)
    # split_val_test_json(train51_studies,
    #                     in_annotation_path=os.path.join(data_dir, 'in', 'lumbar_train51_annotation.json'),
    #                     out_val_annotation_path=os.path.join(data_dir, 'in', 'lumbar_val_annotation.json'),
    #                     out_test_annotation_path=os.path.join(data_dir, 'in', 'lumbar_test_annotation.json'),
    #                     val_ratio=0.5)

    train_dataloader = KeyPointDataLoader(
        train51_studies, train51_annotation, batch_size=5, sagittal_size=(512, 512), num_workers=0, prob_rotate=1.0,
        max_angel=15
    )
    # 设定训练参数
    # train_dataloader = DisDataLoader(
    #     train51_studies, train51_annotation, batch_size=5, num_workers=0, num_rep=1,
    #     prob_rotate=0.5, max_angel=15,
    #     sagittal_size=(64, 64),
    #     transverse_size=(128, 128),
    #     v_k_nearest=8,
    #     d_k_nearest=3,
    #     classifier='ver'
    # )

    for data in train_dataloader:
        pass


