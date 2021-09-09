import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from datasets.augment import rotate_image, rotate_point, rotate
import matplotlib.pylab as plt
from datasets.utils import generate_heatmaps
import itertools
from torch.utils.data.sampler import Sampler
import random
from skimage.exposure import match_histograms
import torch

class H5Dataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """
    def __init__(self, data, phase='train', prob_rotate=0.4,
        max_angle=15, task='localization', use_weak_label=False, batch_size=1, labeled_bs=1):
        self.raw = data['mr']
        self.labels = data['mask']
        self.coords = data['coords']
        self.pixelspacing = data['pixelspacing']
        self.img_size = data['img_size']
        self.phase = phase
        self.random_state = np.random.RandomState(seed=47)
        self.prob_rotate = prob_rotate
        self.max_angle = max_angle
        self.task = task
        self.labeled_num = self.labels.shape[0]
        self.weak_label_num = self.raw.shape[0] - self.labeled_num
        self.use_weak_label = use_weak_label
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs

        n, c, h, w = self.raw.shape
        point_num = self.coords.shape[1]
        if self.task != 'segmentation':
            self.heatmaps = np.zeros((n, point_num, h, w), dtype=np.float32)
            for i in range(n):
                self.heatmaps[i] = generate_heatmaps(self.raw[i, 0, :, :], self.pixelspacing[i], self.coords[i], sigma=3.0)

    def __getitem__(self, index):
        # index = index % self.raw.shape[0]
        raw = np.squeeze(self.raw[index]) # (h, w)

        coords = self.coords[index] # (point_num, 2)
        if self.task == 'localization':
            heatmaps = self.heatmaps[index] # (point_num, h, w)
        elif self.task == 'segmentation':
            assert self.use_weak_label == False
            if self.phase is 'train':
                if index < self.labeled_num:
                    mask = self.labels[index]  # (h, w)
                    mask = mask.astype(np.long)
                else:
                    mask = np.zeros(raw.shape, dtype=np.long)
            else:
                mask = self.labels[index]  # (h, w)
                mask = mask.astype(np.long)
        else:
            heatmaps = self.heatmaps[index]  # (point_num, h, w)
            if self.phase is 'train':
                if index < self.labeled_num:
                    mask = self.labels[index]  # (h, w)
                    mask = mask.astype(np.long)
                else:
                    mask = np.zeros(raw.shape, dtype=np.long)
            else:
                mask = self.labels[index]  # (h, w)
                mask = mask.astype(np.long)

        if self.phase == 'train':
            if self.max_angle > 0 and self.random_state.uniform() <= self.prob_rotate:
                angle = self.random_state.randint(-self.max_angle, self.max_angle)
                raw, coords = rotate(raw, coords, angle=angle, interpolation='bilinear')

                if self.task == 'localization':
                    heatmaps = rotate_image(heatmaps, angle=angle, interpolation='bilinear')
                elif self.task == 'segmentation':
                    mask = rotate_image(mask, angle=angle, interpolation='nearest')
                    mask = mask.astype(np.long)
                else:
                    heatmaps = rotate_image(heatmaps, angle=angle, interpolation='bilinear')
                    mask = rotate_image(mask, angle=angle, interpolation='nearest')
                    mask = mask.astype(np.long)

        raw = np.expand_dims(raw, axis=0)
        if self.task == 'localization':
            return raw.astype(np.float32), heatmaps, coords.round().astype(np.long), \
                   self.pixelspacing[index].astype(np.float32)
        elif self.task == 'segmentation':
            return raw.astype(np.float32), mask, coords.round().astype(np.long), \
                   self.pixelspacing[index].astype(np.float32)
        else:
            return raw.astype(np.float32), mask, heatmaps, coords.round().astype(np.long), \
                   self.pixelspacing[index].astype(np.float32)

    def __len__(self):
        # if self.phase == 'train':
        #     return self.raw.shape[0] * 10
        # else:
        #     return self.raw.shape[0]
        if self.phase == 'train' and self.use_weak_label is False:
            return self.labeled_num
        else:
            return self.raw.shape[0]

    def collate_fn(self, data):
        labeled_data = data[:self.labeled_bs]
        weak_labeled_data = data[self.labeled_bs:]
        rand_ind = self.random_state.randint(0, len(labeled_data))

        template_data = labeled_data[rand_ind]
        template = template_data[0]
        template = np.transpose(template, axes=(1, 2, 0))

        matched_weak_labeled_data = []
        for i in range(len(weak_labeled_data)):
            raw = weak_labeled_data[i][0]
            raw = np.transpose(raw, axes=(1, 2, 0))  # h, w, c
            matched = match_histograms(raw, reference=template, multichannel=True)
            matched = np.transpose(matched, axes=(2, 0, 1))  # c, h, w
            a_weak_labeled_data = (matched, *weak_labeled_data[i][1:])
            matched_weak_labeled_data.append(a_weak_labeled_data)

        data[self.labeled_bs:] = matched_weak_labeled_data

        if self.task == 'localization':
            raws = []
            heatmaps = []
            coords = []
            pixelspacings = []
            for raw, heatmap, coord, pixelspacing in data:
                raws.append(raw.astype(np.float32))
                heatmaps.append(heatmap)
                coords.append(coord.round().astype(np.long))
                pixelspacings.append(pixelspacing.astype(np.float32))
            raws = np.stack(raws, axis=0)
            heatmaps = np.stack(heatmaps, axis=0)
            coords = np.stack(coords, axis=0)
            pixelspacings = np.stack(pixelspacings, axis=0)

            raws = torch.from_numpy(raws)
            heatmaps = torch.from_numpy(heatmaps)
            coords = torch.from_numpy(coords)
            pixelspacings = torch.from_numpy(pixelspacings)
            return raws, heatmaps, coords, pixelspacings
        elif self.task == 'segmentation':
            raws = []
            masks = []
            coords = []
            pixelspacings = []
            for raw, mask, coord, pixelspacing in data:
                raws.append(raw.astype(np.float32))
                masks.append(mask)
                coords.append(coord.round().astype(np.long))
                pixelspacings.append(pixelspacing.astype(np.float32))
            raws = np.stack(raws, axis=0)
            masks = np.stack(masks, axis=0)
            coords = np.stack(coords, axis=0)
            pixelspacings = np.stack(pixelspacings, axis=0)

            raws = torch.from_numpy(raws)
            masks = torch.from_numpy(masks)
            coords = torch.from_numpy(coords)
            pixelspacings = torch.from_numpy(pixelspacings)
            return raws, masks, coords, pixelspacings
        else:
            raws = []
            masks = []
            heatmaps = []
            coords = []
            pixelspacings = []
            for raw, mask, heatmap, coord, pixelspacing in data:
                raws.append(raw.astype(np.float32))
                masks.append(mask)
                heatmaps.append(heatmap)
                coords.append(coord.round().astype(np.long))
                pixelspacings.append(pixelspacing.astype(np.float32))
            raws = np.stack(raws, axis=0)
            masks = np.stack(masks, axis=0)
            heatmaps = np.stack(heatmaps, axis=0)
            coords = np.stack(coords, axis=0)
            pixelspacings = np.stack(pixelspacings, axis=0)

            raws = torch.from_numpy(raws)
            masks = torch.from_numpy(masks)
            heatmaps = torch.from_numpy(heatmaps)
            coords = torch.from_numpy(coords)
            pixelspacings = torch.from_numpy(pixelspacings)
            return raws, masks, heatmaps, coords, pixelspacings


    @staticmethod
    def _transform_image(dataset, transformer):
        return transformer(dataset)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    Reference: Shape-aware Semi-supervised 3D Semantic Segmentation for Medical Images, MICCAI 2020.
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def worker_init_fn(worker_id):
    worker_seed = 0 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def load_data(filePath, prob_rotate=0.4,
        max_angle=15, batch_size=2, labeled_bs=1, num_workers=0, task='localization', use_weak_label=False,
              use_histograms_match=False):
    f = h5py.File(filePath, 'r')
    train_data = f['train']
    val_data = f['val']
    test_data = f['test']
    train_set = H5Dataset(train_data, phase='train', prob_rotate=prob_rotate, max_angle=max_angle, task=task,
                          use_weak_label=use_weak_label, batch_size=batch_size, labeled_bs=labeled_bs)
    val_set = H5Dataset(val_data, phase='val', prob_rotate=prob_rotate, max_angle=max_angle, task=task,
                        use_weak_label=False, batch_size=batch_size, labeled_bs=labeled_bs)
    test_set = H5Dataset(test_data, phase='test', prob_rotate=prob_rotate, max_angle=max_angle, task=task,
                         use_weak_label=False, batch_size=batch_size, labeled_bs=labeled_bs)

    labeled_num = train_set.labeled_num
    weak_label_num = train_set.weak_label_num
    labeled_idxs = list(range(labeled_num))
    unlabeled_idxs = list(range(labeled_num, labeled_num + weak_label_num))

    generator = torch.Generator()
    generator.manual_seed(0)

    if use_weak_label:
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        if use_histograms_match:
            training_data_loader = DataLoader(dataset=train_set, batch_sampler=batch_sampler, num_workers=num_workers,
                                          pin_memory=False, worker_init_fn=worker_init_fn,
                                              generator=generator, collate_fn=train_set.collate_fn)
        else:
            training_data_loader = DataLoader(dataset=train_set, batch_sampler=batch_sampler, num_workers=num_workers,
                                              pin_memory=False, worker_init_fn=worker_init_fn, generator=generator)
    else:
        training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                     shuffle=True, worker_init_fn=worker_init_fn, generator=generator, drop_last=True)


    val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, pin_memory=False,
                                      shuffle=False)

    # val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=batch_size, pin_memory=False,
    #                              shuffle=True)

    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, pin_memory=False,
                                     shuffle=False)
    return training_data_loader, val_data_loader, testing_data_loader, f

if __name__ == '__main__':
    inDir = '/public/pangshumao/data/Spine_Localization_PIL/in/h5py'
    filePath = os.path.join(inDir, 'data_fold1.h5')

    try:
        train_data_loader, val_data_loader, test_data_loader, f = load_data(filePath, batch_size=4, labeled_bs=2,
                                                                            task='multi-task', use_weak_label=True,
                                                                            use_histograms_match=True)
        batch_num = len(train_data_loader)
        for i, t in enumerate(train_data_loader):
            print(i)
            mr,  mask, coords, heatmaps, pixelspacings = t
            print(mr.shape)

            image = mr.detach().numpy()
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                                sharex=True, sharey=True)

            for aa in (ax1, ax2, ax3):
                aa.set_axis_off()

            ax1.imshow(image[0, 0, :, :], cmap='gray')
            ax1.set_title('labeled 1')
            ax2.imshow(image[1, 0, :, :], cmap='gray')
            ax2.set_title('labeled 2')
            ax3.imshow(image[2, 0, :, :], cmap='gray')
            ax3.set_title('Matched')

            plt.tight_layout()
            plt.show()
    finally:
        f.close()

