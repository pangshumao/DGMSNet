import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
from datasets.augment import resize_point
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math

def test_read_h5():
    data_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/h5py'
    f = h5py.File(os.path.join(data_dir, 'data_fold1.h5'), 'r')
    train_mr = f['train']['mr']
    train_label = f['train']['mask']
    print(train_mr.shape)
    print(train_label.shape)
    plt.imshow(train_mr[5, 0, :, :], cmap='gray')
    plt.show()

def normalization(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())

def pad_img(img, img_size):
    img_d, img_h, img_w = img.shape
    target_d, target_h, target_w = img_size[0], img_size[1], img_size[2]

    padded = np.zeros((target_d, target_h, target_w), dtype=img.dtype)

    padded[(target_d - img_d) // 2: (target_d - img_d) // 2 + img_d, (target_h - img_h) // 2:(target_h - img_h) // 2 + img_h,
    (target_w - img_w) // 2:(target_w - img_w) // 2 + img_w] = img

    return padded

def make_h5(data_dir, h5_group, case_inds, img_size, weak_label_num=0):
    weak_mr_dir = os.path.join(data_dir, 'in', 'weak_supervised_MR')
    mr_dir = os.path.join(data_dir, 'in', 'nii', 'MR')
    mask_dir = os.path.join(data_dir, 'in', 'nii', 'processed_mask')
    keypoints_dir = os.path.join(data_dir, 'in', 'keypoints')

    sample_num = case_inds.size

    # For data
    flag = True
    for case_ind in case_inds:
        print('processing case%d' % case_ind)
        if case_ind < 216:
            mr = nib.load(os.path.join(mr_dir, 'Case' + str(case_ind) + '.nii.gz')).get_data()  # [h, w, 1]
            mask = nib.load(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz')).get_data()  # [h, w, 1]
            mr = np.rot90(mr[:, :, 0])
            mask = np.rot90(mask[:, :, 0])
            mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant',
                                    anti_aliasing=False).astype(np.int)
            mask = mask[None, :, :]
        else:
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(weak_mr_dir, 'Case' + str(case_ind) + '.dcm'))
            image = reader.Execute()
            mr = sitk.GetArrayFromImage(image)  # 1, h, w
            mr = mr[0, :, :]
        kp_npz = np.load(os.path.join(keypoints_dir, 'Case' + str(case_ind) + '.npz'))
        coords = kp_npz['coords'] # (10, 2)
        original_img_size = kp_npz['img_size'] # (2, ), [height, width]
        pixelspacing = kp_npz['pixelspacing']  # (2, ), [pixel_size_w, pixel_size_h]
        pixelspacing[0] = pixelspacing[0] * original_img_size[1] / img_size[1]
        pixelspacing[1] = pixelspacing[1] * original_img_size[0] / img_size[0]


        mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
        coords = resize_point(coords, in_img_size=original_img_size, out_img_size=img_size)

        # plt.subplot(121)
        # plt.imshow(mr, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(mask, cmap='gray')
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

            if case_ind < 216:
                h5_group['mask'].resize(h5_group['mask'].shape[0] + mask.shape[0], axis=0)
                h5_group['mask'][-mask.shape[0]:] = mask

            h5_group['coords'].resize(h5_group['coords'].shape[0] + coords.shape[0], axis=0)
            h5_group['coords'][-coords.shape[0]:] = coords

            h5_group['pixelspacing'].resize(h5_group['pixelspacing'].shape[0] + pixelspacing.shape[0], axis=0)
            h5_group['pixelspacing'][-pixelspacing.shape[0]:] = pixelspacing

            h5_group['img_size'].resize(h5_group['img_size'].shape[0] + original_img_size.shape[0], axis=0)
            h5_group['img_size'][-original_img_size.shape[0]:] = original_img_size


if __name__ == '__main__':
    # test_read_h5()

    data_dir = '/public/pangshumao/data/Spine_Localization_PIL'

    img_size = [780, 780]
    weak_label_num = 201

    for fold_ind in range(1, 6):
        print('processing fold:%d...............................................' % fold_ind)
        split_data = np.load(os.path.join(data_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
        train_ind = split_data['train_ind']
        val_ind = split_data['val_ind']
        test_ind = split_data['test_ind']

        weak_label_ind = np.array(range(216, 417))
        train_ind = np.concatenate((train_ind, weak_label_ind), axis=0)

        f = h5py.File(os.path.join(data_dir, 'in', 'h5py', 'data_fold' + str(fold_ind) + '.h5'), 'w')
        g_train = f.create_group('train')
        g_val = f.create_group('val')
        g_test = f.create_group('test')

        make_h5(data_dir, g_train, train_ind, img_size, weak_label_num)
        make_h5(data_dir, g_val, val_ind, img_size)
        make_h5(data_dir, g_test, test_ind, img_size)

        f.close()
    print('Job done!!!')






