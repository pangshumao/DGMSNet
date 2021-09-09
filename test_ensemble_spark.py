import sys
import time
import os
import torch
sys.path.append('nn_tools/')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from skimage import transform
from scipy import ndimage
from datasets.augment import resize_point
from torch.nn import functional as F
from networks.evaluation import cal_acc, dice_all_class
import matplotlib.pyplot as plt
from networks.evaluation import cal_distances
from datasets.utils import expand_as_one_hot
import SimpleITK as sitk
import pydicom
from datasets.utils import pad_image_coords_to_square, unpad_square_coords, unpad_square_image_coords

spine_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
def cal_keypoints(mask, class_num):
    coords = np.zeros((class_num - 1, 2), dtype=np.float32)
    for i in range(1, class_num):
        pos = np.argwhere(mask == i)
        count = pos.shape[0]
        center_h, center_w = pos.sum(0) / (count + 1e-6)
        center_h = round(center_h)
        center_w = round(center_w)
        coords[i - 1, :] = np.array([center_w, center_h])  # x, y
    return coords

def construct_identifier(beta, args):
    identifier = args.task + '_' + args.model + '_' + args.optimizer + '_lr_' + str(
        args.learning_rate) + '_beta_' + str(beta)
    out_identifier = args.task + '_' + args.model + '_' + args.optimizer + '_lr_' + str(
        args.learning_rate)

    if args.os != 16:
        identifier = identifier + '_os_' + str(args.os)
        out_identifier = out_identifier + '_os_' + str(args.os)

    if args.use_weak_label:
        identifier = identifier + '_use_weak_label'
        out_identifier = out_identifier + '_use_weak_label'

    if args.batch_size > 2:
        identifier = identifier + '_batch_size_' + str(args.batch_size)
        out_identifier = out_identifier + '_batch_size_' + str(args.batch_size)

    if args.epochs > 100:
        identifier = identifier + '_epochs_' + str(args.epochs)
        out_identifier = out_identifier + '_epochs_' + str(args.epochs)

    if args.weight_decay != 1e-4:
        identifier = identifier + '_weight_decay_' + str(args.weight_decay)
        out_identifier = out_identifier + '_weight_decay_' + str(args.weight_decay)

    if not args.use_scheduler:
        identifier = identifier + '_no-use_scheduler'
        out_identifier = out_identifier + '_no-use_scheduler'

    out_identifier = out_identifier + '_ensemble_type_' + args.ensemble_type
    return identifier, out_identifier

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str, default='DeepLabv3_plus_lil',
                        help="the model name, "
                             "DeepLabv3_plus_lil")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/DGMSNet/Spinal_disease_dataset',
                        help="the data dir")

    parser.add_argument("--task", type=str, default='multi-task',
                        help="multi-task")

    parser.add_argument("--ensemble_type", type=str, default='model-selection',
                        help="model-selection, majority-voting, model-selection-mv")

    parser.add_argument('--batch_size', type=int, default=3,
                        help="The batch size")

    parser.add_argument("--epochs", type=int, default=100,
                        help="max number of epochs")

    parser.add_argument("--use_weak_label", dest='use_weak_label', action='store_true',
                        help="use use_weak_label")

    parser.add_argument("--no-use_weak_label", dest='use_weak_label', action='store_false',
                        help="without using use_weak_label")

    parser.set_defaults(use_weak_label=False)

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, MSELoss, CrossEntropyLoss.")

    parser.add_argument('--beta', type=float, default=60.0, nargs='+',
                        help="the localization loss weight")

    parser.add_argument("--use_scheduler", dest='use_scheduler', action='store_true',
                        help="use scheduler")

    parser.add_argument("--no-use_scheduler", dest='use_scheduler', action='store_false',
                        help="without using scheduler")

    parser.set_defaults(use_scheduler=True)

    parser.add_argument('--os', type=int, default=16,
                        help="the output stride of encoder.")

    parser.add_argument("--devices", type=str, default='cuda:0', nargs='+',
                        help="which gpus to use")

    parser.add_argument("--fold_ind", type=int, default=1,
                        help="fold index, 1, 2.")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam or SGD")

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='the weight decay')

    parser.add_argument('--learning_rate', type=float, default=3e-3,
                        help="The initial learning rate")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    img_size = (512, 512)
    data_dir = args.data_dir
    fold_ind = args.fold_ind
    fold_dir = os.path.join(data_dir, 'model', 'fold' + str(fold_ind))
    fold_ind_data = np.load(os.path.join(data_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
    train_ind = fold_ind_data['train_ind']
    val_ind = fold_ind_data['val_ind']
    test_ind = fold_ind_data['test_ind']

    mr_dir = os.path.join(data_dir, 'in', 'MR')
    kp_dir = os.path.join(data_dir, 'in', 'keypoints')
    mask_dir = os.path.join(data_dir, 'in', 'mask')

    beta_num = len(args.beta)
    test_num = test_ind.shape[0]
    dices = np.zeros(test_num, dtype=np.float32)
    min_dists = np.ones(test_num, dtype=np.float32) * 100000
    if args.ensemble_type == 'model-selection':
        for beta_ind, beta in enumerate(args.beta):
            identifier, out_identifier = construct_identifier(beta, args)
            model_dir = os.path.join(fold_dir, identifier)
            out_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), out_identifier)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            model_path = os.path.join(model_dir, 'best.module')
            # model_path = os.path.join(model_dir, 'last.module')
            # model_path = os.path.join(model_dir, 'lowest_loss.module')

            model = torch.load(model_path, map_location='cpu')
            model.devices = [args.devices]
            model.connection = args.connection
            model = model.to(args.devices[0])

            target_coords_list = []
            pred_coords_list = []
            pixelspacing_list = []
            dice_list = []
            model.eval()

            for i, case_ind in enumerate(test_ind):
                # print('processing case%d' % case_ind)
                mr_sitk_image = sitk.ReadImage(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                mr = sitk.GetArrayFromImage(mr_sitk_image)  # 1, h, w
                mr = mr[0, :, :]

                pydicom_file = pydicom.read_file(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                pixelspacing = np.array([float(pydicom_file.PixelSpacing[0]), float(pydicom_file.PixelSpacing[1])])

                mask_image = sitk.ReadImage(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
                target_mask = sitk.GetArrayFromImage(mask_image)

                kp_npz = np.load(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'))

                pixelspacing_list.append(pixelspacing)

                target_coords = kp_npz['coords']  # (10, 2)
                target_coords_list.append(target_coords)
                original_img_size = mr.shape

                mr = pad_image_coords_to_square(mr)
                original_square_img_size = mr.shape

                mr_resize = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant',
                                             anti_aliasing=False) # resampled space

                mr_resize = mr_resize[None, None, :, :]
                if isinstance(args.devices, list) or isinstance(args.devices, tuple):
                    mr_resize = torch.from_numpy(mr_resize).to(args.devices[0])
                else:
                    mr_resize = torch.from_numpy(mr_resize).to(args.devices)

                logit, pred_coords = model(mr_resize)

                logit = F.interpolate(logit, size=original_square_img_size, mode='bilinear', align_corners=True)
                prob = F.softmax(logit, dim=1)
                seg = torch.argmax(prob, dim=1).squeeze(dim=0).detach().to('cpu').numpy()
                seg = unpad_square_image_coords(seg, original_img_size)  # (h, w)
                pred_seg_coords = cal_keypoints(seg, class_num=logit.shape[1])

                pred_coords = resize_point(pred_coords.squeeze().detach().to('cpu').numpy(), in_img_size=img_size,
                                           out_img_size=original_square_img_size)  # original padding space

                pred_coords = unpad_square_coords(pred_coords, original_img_size)

                pred_coords = pred_coords[None, :, :]
                pred_seg_coords = pred_seg_coords[None, :, :]

                dist = cal_distances(pred_coords, pred_seg_coords, pixel_spacings=pixelspacing[None, :])
                dist = np.mean(dist)
                if dist < min_dists[i]:
                    min_dists[i] = dist
                    dice = dice_all_class(seg, target_mask, class_num=logit.shape[1])
                    dices[i] = dice

                    seg_sitk = sitk.GetImageFromArray(seg.astype(np.int16))
                    seg_sitk.SetSpacing(pixelspacing)
                    seg_sitk.SetOrigin(mr_sitk_image.GetOrigin())
                    sitk.WriteImage(seg_sitk, os.path.join(out_dir, 'seg_' + str(case_ind) + '.nii.gz'))

        for case_ind, dice in zip(test_ind, dices):
            print('fold%d Case%d dice = %.2f' % (fold_ind, case_ind, dice * 100))
        mean_dice = np.mean(dices)
        print('fold%d mean dice = %.4f' % (fold_ind, mean_dice))
        end_time = time.time()
        mean_time = (end_time - start_time) / test_num
        print('task completed, {} seconds used per case'.format(mean_time))
    elif args.ensemble_type == 'majority-voting':
        for i, case_ind in enumerate(test_ind):
            for beta_ind, beta in enumerate(args.beta):
                identifier, out_identifier = construct_identifier(beta, args)
                model_dir = os.path.join(fold_dir, identifier)
                out_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), out_identifier)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                model_path = os.path.join(model_dir, 'best.module')
                # model_path = os.path.join(model_dir, 'last.module')
                # model_path = os.path.join(model_dir, 'lowest_loss.module')

                model = torch.load(model_path, map_location='cpu')
                model.devices = [args.devices]
                model.connection = args.connection
                model = model.to(args.devices[0])

                target_coords_list = []
                pred_coords_list = []
                pixelspacing_list = []
                dice_list = []
                model.eval()

                with torch.no_grad():
                    # print('processing case%d' % case_ind)
                    mr_sitk_image = sitk.ReadImage(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                    mr = sitk.GetArrayFromImage(mr_sitk_image)  # 1, h, w
                    mr = mr[0, :, :]

                    pydicom_file = pydicom.read_file(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                    pixelspacing = np.array([float(pydicom_file.PixelSpacing[0]), float(pydicom_file.PixelSpacing[1])])

                    mask_image = sitk.ReadImage(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
                    target_mask = sitk.GetArrayFromImage(mask_image)

                    kp_npz = np.load(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'))

                    pixelspacing_list.append(pixelspacing)

                    target_coords = kp_npz['coords']  # (10, 2)
                    target_coords_list.append(target_coords)
                    original_img_size = mr.shape

                    mr = pad_image_coords_to_square(mr)
                    original_square_img_size = mr.shape
                    mr_resize = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant',
                                                 anti_aliasing=False)  # resampled space

                    mr_resize = mr_resize[None, None, :, :]
                    if isinstance(args.devices, list) or isinstance(args.devices, tuple):
                        mr_resize = torch.from_numpy(mr_resize).to(args.devices[0])
                    else:
                        mr_resize = torch.from_numpy(mr_resize).to(args.devices)
                    logit, pred_coords = model(mr_resize)
                    logit = F.interpolate(logit, size=original_square_img_size, mode='bilinear', align_corners=True)
                    prob = F.softmax(logit, dim=1)
                    seg = torch.argmax(prob, dim=1).squeeze().detach().to('cpu').numpy()

                    seg = unpad_square_image_coords(seg, original_img_size)  # (h, w)
                    seg = torch.from_numpy(seg[None, :, :]).to(mr_resize.device)
                    if beta_ind == 0:
                        seg_one_hot = expand_as_one_hot(seg, prob.shape[1]).to('cpu').numpy() # (1, class_num, h, w)
                    else:
                        seg_one_hot += expand_as_one_hot(seg, prob.shape[1]).to('cpu').numpy() # (1, class_num, h, w)

            seg = np.argmax(seg_one_hot, axis=1).squeeze(axis=0)    # (h, w)

            dice = dice_all_class(seg, target_mask, class_num=logit.shape[1])
            dices[i] = dice
            seg_sitk = sitk.GetImageFromArray(seg.astype(np.int16))
            seg_sitk.SetSpacing(pixelspacing)
            seg_sitk.SetOrigin(mr_sitk_image.GetOrigin())
            sitk.WriteImage(seg_sitk, os.path.join(out_dir, 'seg_' + str(case_ind) + '.nii.gz'))
            print('fold%d Case%d dice = %.2f' % (fold_ind, case_ind, dice * 100))
        mean_dice = np.mean(dices)
        print('fold%d mean dice = %.4f' % (fold_ind, mean_dice))
        end_time = time.time()
        mean_time = (end_time - start_time) / test_num
        print('task completed, {} seconds used per case'.format(mean_time))
    elif args.ensemble_type == 'model-selection-mv':
        for i, case_ind in enumerate(test_ind):
            dists = np.zeros(beta_num, dtype=np.float32)
            ms_segs = []
            for beta_ind, beta in enumerate(args.beta):
                identifier, out_identifier = construct_identifier(beta, args)
                model_dir = os.path.join(fold_dir, identifier)
                out_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), out_identifier)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                model_path = os.path.join(model_dir, 'best.module')
                # model_path = os.path.join(model_dir, 'last.module')
                # model_path = os.path.join(model_dir, 'lowest_loss.module')

                model = torch.load(model_path, map_location='cpu')
                model.devices = [args.devices]
                model.connection = args.connection
                model = model.to(args.devices[0])

                target_coords_list = []
                pred_coords_list = []
                pixelspacing_list = []
                dice_list = []
                model.eval()

                # print('processing case%d' % case_ind)
                mr_sitk_image = sitk.ReadImage(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                mr = sitk.GetArrayFromImage(mr_sitk_image)  # 1, h, w
                mr = mr[0, :, :]

                pydicom_file = pydicom.read_file(os.path.join(mr_dir, 'Case' + str(case_ind) + '.dcm'))
                pixelspacing = np.array([float(pydicom_file.PixelSpacing[0]), float(pydicom_file.PixelSpacing[1])])

                mask_image = sitk.ReadImage(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
                target_mask = sitk.GetArrayFromImage(mask_image)

                kp_npz = np.load(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'))

                pixelspacing_list.append(pixelspacing)

                target_coords = kp_npz['coords']  # (10, 2)
                target_coords_list.append(target_coords)
                original_img_size = mr.shape

                mr = pad_image_coords_to_square(mr)
                original_square_img_size = mr.shape
                mr_resize = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant',
                                             anti_aliasing=False)  # resampled space

                mr_resize = mr_resize[None, None, :, :]
                if isinstance(args.devices, list) or isinstance(args.devices, tuple):
                    mr_resize = torch.from_numpy(mr_resize).to(args.devices[0])
                else:
                    mr_resize = torch.from_numpy(mr_resize).to(args.devices)

                logit, pred_coords = model(mr_resize)

                logit = F.interpolate(logit, size=original_square_img_size, mode='bilinear', align_corners=True)
                prob = F.softmax(logit, dim=1)
                # for majority-voting
                seg = torch.argmax(prob, dim=1).squeeze().detach().to('cpu').numpy()
                seg = unpad_square_image_coords(seg, original_img_size) # (h, w)
                seg = torch.from_numpy(seg[None, :, :]).to(mr_resize.device)
                if beta_ind == 0:
                    seg_one_hot = expand_as_one_hot(seg, prob.shape[1]).to('cpu').numpy() # (1, class_num, h, w)
                else:
                    seg_one_hot += expand_as_one_hot(seg, prob.shape[1]).to('cpu').numpy() # (1, class_num, h, w)

                # for model-selection
                ms_seg = torch.argmax(prob, dim=1).squeeze(dim=0).detach().to('cpu').numpy()
                ms_seg = unpad_square_image_coords(ms_seg, original_img_size)  # (h, w)
                pred_seg_coords = cal_keypoints(ms_seg, class_num=logit.shape[1])
                ms_segs.append(ms_seg)

                pred_coords = resize_point(pred_coords.squeeze().detach().to('cpu').numpy(), in_img_size=img_size,
                                           out_img_size=original_square_img_size)  # original padding space
                pred_coords = unpad_square_coords(pred_coords, original_img_size)

                pred_coords = pred_coords[None, :, :]
                pred_seg_coords = pred_seg_coords[None, :, :]

                dist = cal_distances(pred_coords, pred_seg_coords, pixel_spacings=pixelspacing[None, :])
                dist = np.mean(dist)
                dists[beta_ind] = dist

            threshold = dists.mean() / 4.0
            std = dists.std()
            if std > threshold:
                # model-selection
                min_dist_ind = np.argmin(dists)
                seg = ms_segs[min_dist_ind]
                flag = 'model-selection'
            else:
                # majority-voting
                seg = np.argmax(seg_one_hot, axis=1).squeeze(axis=0)  # (h, w)
                flag = 'majority-voting'

            dice = dice_all_class(seg, target_mask, class_num=logit.shape[1])
            dices[i] = dice
            seg_sitk = sitk.GetImageFromArray(seg.astype(np.int16))
            seg_sitk.SetSpacing(pixelspacing)
            seg_sitk.SetOrigin(mr_sitk_image.GetOrigin())
            sitk.WriteImage(seg_sitk, os.path.join(out_dir, 'seg_' + str(case_ind) + '.nii.gz'))

            print('fold%d Case%d %s dist_std = %.4f threshold = %.4f dice = %.2f' % (fold_ind, case_ind, flag, std, threshold, dice * 100))
        mean_dice = np.mean(dices)
        print('fold%d mean dice = %.4f' % (fold_ind, mean_dice))
        end_time = time.time()
        mean_time = (end_time - start_time) / test_num
        print('task completed, {} seconds used per case'.format(mean_time))




