import sys
import time
import os
import torch
sys.path.append('nn_tools/')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import nibabel as nib
from skimage import transform
from scipy import ndimage
from datasets.augment import resize_point
from torch.nn import functional as F
from networks.evaluation import cal_acc, dice_all_class, cal_distance_per_image, cal_distances
import matplotlib.pyplot as plt
from datasets.utils import mask2coords


spine_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str, default='DeepLabv3_plus_lil',
                        help="the model name, "
                             "DeepLabv3_plus"
                             "DeepLabv3_plus_gcn, "
                             "DeepLabv3_plus_lil")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/DGMSNet/In_house_dataset',
                        help="the data dir")

    parser.add_argument("--task", type=str, default='multi-task',
                        help="localization, segmentation, or multi-task")

    parser.add_argument("--pre_trained", dest='pre_trained', action='store_true',
                        help="For DeepLabv3_plus_gcn.")

    parser.add_argument("--no-pre_trained", dest='pre_trained', action='store_false',
                        help="For DeepLabv3_plus_gcn.")

    parser.set_defaults(pre_trained=False)

    parser.add_argument('--pre_trained_identifier', type=str, default='segmentation_DeepLabv3_plus_Adam_lr_0.005_'
                                                                      'CrossEntropyLoss_batch_size_4_epochs_200',
                        help='It is available when pre_trained is True.')

    parser.add_argument('--gcn_num', type=int, default=3,
                        help='The gcn number for DeepLabv3_plus_gcn model')

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

    parser.add_argument('--beta', type=float, default=60.0,
                        help="the localization loss weight or the deep supervised loss weight for gcn model.")

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
                        help="fold index, 1, 2, 3, 4, 5.")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam or SGD")

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='the weight decay')

    parser.add_argument('--learning_rate', type=float, default=3e-3,
                        help="The initial learning rate")

    parser.add_argument("--seed", type=int, default=0,
                        help="init seed")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)


    data_dir = args.data_dir
    fold_ind = args.fold_ind
    fold_dir = os.path.join(data_dir, 'model', 'fold' + str(fold_ind))
    fold_ind_data = np.load(os.path.join(data_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
    train_ind = fold_ind_data['train_ind']
    val_ind = fold_ind_data['val_ind']
    test_ind = fold_ind_data['test_ind']

    mr_dir = os.path.join(data_dir, 'in', 'nii', 'MR')
    kp_dir = os.path.join(data_dir, 'in', 'keypoints')
    mask_dir = os.path.join(data_dir, 'in', 'nii', 'processed_mask')

    if args.task == 'multi-task':
        identifier = args.task + '_' + args.model + '_' + args.optimizer + '_lr_' + str(
            args.learning_rate) + '_beta_' + str(args.beta)
    else:
        identifier = args.task + '_' + args.model + '_' + args.optimizer + '_lr_' + str(args.learning_rate) + '_' + args.loss

    if args.os != 16:
        identifier = identifier + '_os_' + str(args.os)

    if args.use_weak_label:
        identifier = identifier + '_use_weak_label'

    if args.task == 'localization':
        out_channels = 10
    elif args.task == 'segmentation':
        out_channels = 11
        if args.batch_size > 2:
            identifier = identifier + '_batch_size_' + str(args.batch_size)

    if args.task == 'multi-task':
        if args.batch_size > 2:
            identifier = identifier + '_batch_size_' + str(args.batch_size)

    if args.epochs != 100:
        identifier = identifier + '_epochs_' + str(args.epochs)

    if args.weight_decay != 1e-4:
        identifier = identifier + '_weight_decay_' + str(args.weight_decay)

    if not args.use_scheduler:
        identifier = identifier + '_no-use_scheduler'

    if args.model == 'DeepLabv3_plus_gcn':
        identifier = identifier + '_beta_' + str(args.beta)
        if args.gcn_num != 3:
            identifier += '_gcn_num_' + str(args.gcn_num)
        if args.pre_trained:
            identifier += '_pre_trained'

    if args.seed != 0:
        identifier += '_seed_' + str(args.seed)

    model_dir = os.path.join(fold_dir, identifier)
    out_dir = os.path.join(data_dir, 'out', 'fold' + str(fold_ind), identifier)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(model_dir, 'best.module')
    # model_path = os.path.join(model_dir, 'last.module')
    # model_path = os.path.join(model_dir, 'lowest_loss.module')

    model = torch.load(model_path, map_location='cpu')
    if args.model == 'DeepLabv3_plus_gcn':
        gpu_index = args.devices[0].split(':')[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        model = model.cuda()
    else:
        model.devices = [args.devices]
        model = model.to(args.devices[0])

    # model_path = os.path.join(model_dir, 'best.weight')
    # state = torch.load(model_path)
    # model.load_state_dict(state)

    img_size = (780, 780)
    target_coords_list = []
    pred_coords_list = []
    seg_coords_list = []
    pixelspacing_list = []
    pred_dists_list = []
    seg_dists_list = []
    dice_list = []
    model.eval()
    test_num = test_ind.shape[0]
    for case_ind in test_ind:
        # print('processing case%d' % case_ind)
        nii = nib.load(os.path.join(mr_dir, 'Case' + str(case_ind) + '.nii.gz'))
        mr = nii.get_data()  # [h, w, 1]
        kp_npz = np.load(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'))

        mask_nii = nib.load(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
        target_mask = mask_nii.get_data()  # [h, w, 1]
        target_mask = target_mask[:, :, 0]

        hdr = nii.header
        pixel_size_h, pixel_size_w = hdr['pixdim'][1:3]
        pixelspacing = np.array([pixel_size_w, pixel_size_h])
        pixelspacing_list.append(pixelspacing)

        mr = np.rot90(mr[:, :, 0])

        target_coords = kp_npz['coords']  # (10, 2)
        target_coords_list.append(target_coords)
        original_img_size = mr.shape

        mr_resize = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant',
                                     anti_aliasing=False) # resampled space

        mr_resize = mr_resize[None, None, :, :]
        if args.model == 'DeepLabv3_plus_gcn':
            mr_resize = torch.from_numpy(mr_resize).cuda()
        else:
            if isinstance(args.devices, list) or isinstance(args.devices, tuple):
                mr_resize = torch.from_numpy(mr_resize).to(args.devices[0])
            else:
                mr_resize = torch.from_numpy(mr_resize).to(args.devices)
        if args.task == 'localization':
            pred_coords, pred_heatmaps = model(mr_resize, return_more=True) # resampled space

            pred_coords = resize_point(pred_coords.squeeze().detach().to('cpu').numpy(), in_img_size=img_size,
                                       out_img_size=original_img_size) # original space
            pred_coords_list.append(pred_coords)
            pred_heatmaps = F.interpolate(pred_heatmaps, size=original_img_size, mode='bilinear', align_corners=True)
            pred_heatmaps = pred_heatmaps.squeeze().detach().to('cpu').numpy()

            # acc = cal_acc(pred_coords[None, :, :], target_coords[None, :, :], pixelspacing[None, :], max_dist=6.0)
            #
            # if acc < 1:
            #     print('Case%d acc = %.4f' % (case_ind, acc))
            #     plt.figure(figsize=(10, 8))
            #     plt.subplot(121)
            #     plt.imshow(mr, cmap='gray')
            #     for i in range(len(spine_names)):
            #         plt.scatter(x=pred_coords[i, 0], y=pred_coords[i, 1], c='r')
            #         plt.text(x=pred_coords[i, 0] + 6, y=pred_coords[i, 1] + 4, s=spine_names[i], c='w')
            #     plt.title('Case%d prediction' % case_ind)
            #     plt.axis('off')
            #
            #     plt.subplot(122)
            #     plt.imshow(mr, cmap='gray')
            #     for i in range(len(spine_names)):
            #         plt.scatter(x=target_coords[i, 0], y=target_coords[i, 1], c='r')
            #         plt.text(x=target_coords[i, 0] + 6, y=target_coords[i, 1] + 4, s=spine_names[i], c='w')
            #     plt.title('Case%d GT' % case_ind)
            #     plt.axis('off')
            #     plt.show()
        elif args.task == 'segmentation':
            if args.model == 'DeepLabv3_plus_gcn':
                logit, fea_logit = model(mr_resize)

                # fea_logit = F.interpolate(fea_logit, size=original_img_size, mode='bilinear', align_corners=True)
                # fea_prob = F.softmax(fea_logit, dim=1)
                # fea_prob = fea_prob.squeeze(dim=0).detach().to('cpu').numpy() # (11, h, w)

            else:
                logit = model(mr_resize, task='segmentation')
            logit = F.interpolate(logit, size=original_img_size, mode='bilinear', align_corners=True)
            prob = F.softmax(logit, dim=1)
            seg = torch.argmax(prob, dim=1).squeeze(dim=0).detach().to('cpu').numpy()  # (h, w)
            seg = ndimage.rotate(seg, angle=-90)


            # prob_np = prob.squeeze(dim=0).detach().to('cpu').numpy() # (11, h, w)
            # for c in range(fea_prob.shape[0]):
            #     plt.subplot(131)
            #     plt.imshow(ndimage.rotate(target_mask, angle=90))
            #     plt.title('mask')
            #
            #     plt.subplot(132)
            #     plt.imshow(fea_prob[c])
            #     plt.title('fea_prob')
            #
            #     plt.subplot(133)
            #     plt.imshow(prob_np[c])
            #     plt.title('prob')
            #     plt.show()

            # plt.subplot(121)
            # plt.imshow(target_mask)
            # plt.subplot(122)
            # plt.imshow(seg)
            # plt.show()

            dice = dice_all_class(seg, target_mask, class_num=logit.shape[1])
            print('Case%d dice = %.4f' % (case_ind, dice))
            dice_list.append(dice)
            seg = seg[:,:,None]
            segNii = nib.Nifti1Image(seg.astype(np.int16), affine=nii.affine)
            nib.save(segNii, os.path.join(out_dir, 'seg_' + str(case_ind) + '.nii.gz'))
        else: # multi-task
            logit, pred_coords = model(mr_resize)

            logit = F.interpolate(logit, size=original_img_size, mode='bilinear', align_corners=True)
            prob = F.softmax(logit, dim=1)
            seg = torch.argmax(prob, dim=1).squeeze(dim=0).detach().to('cpu').numpy()  # (h, w)
            seg_coords = mask2coords(seg, class_num=11)
            seg = ndimage.rotate(seg, angle=-90)

            # plt.subplot(121)
            # plt.imshow(target_mask)
            # plt.subplot(122)
            # plt.imshow(seg)
            # plt.show()

            dice = dice_all_class(seg, target_mask, class_num=logit.shape[1])

            dice_list.append(dice)
            seg = seg[:, :, None]
            segNii = nib.Nifti1Image(seg.astype(np.int16), affine=nii.affine)
            nib.save(segNii, os.path.join(out_dir, 'seg_' + str(case_ind) + '.nii.gz'))

            pred_coords = resize_point(pred_coords.squeeze().detach().to('cpu').numpy(), in_img_size=img_size,
                                       out_img_size=original_img_size)  # original space

            pred_dists = cal_distance_per_image(pred_coords, target_coords, pixelspacing)
            seg_dists = cal_distance_per_image(seg_coords, target_coords, pixelspacing)
            pred_dists_list.append(pred_dists)
            seg_dists_list.append(seg_dists)

            pred_coords_list.append(pred_coords)
            seg_coords_list.append(seg_coords)

            pred_acc = cal_acc(pred_coords[None], target_coords[None], pixel_spacings=pixelspacing[None], max_dist=6.0)
            seg_acc = cal_acc(seg_coords[None], target_coords[None], pixel_spacings=pixelspacing[None], max_dist=6.0)
            print('Case%d dice = %.2f, seg_acc = %.2f, pred_acc = %.2f, '
                  'seg_dist = %.2f mm, pred_dist = %.2f mm' % (case_ind, dice * 100,
                                                               seg_acc * 100,
                                                               pred_acc * 100,
                                                               np.mean(seg_dists),
                                                               np.mean(pred_dists)))
            np.savez(os.path.join(out_dir, 'keypoints_case' + str(case_ind) + '.npz'), pred_coords=pred_coords,
                     seg_coords=seg_coords, pred_dists=pred_dists, seg_dists=seg_dists, pred_acc=pred_acc,
                     seg_acc=seg_acc, pixelspacing=pixelspacing)

    if args.task == 'localization':
        total_target_coords = np.stack(target_coords_list, axis=0)
        total_pred_coords = np.stack(pred_coords_list, axis=0)
        total_pixelspacings = np.stack(pixelspacing_list, axis=0)
        mean_acc = cal_acc(total_pred_coords, total_target_coords, total_pixelspacings, max_dist=6.0)
        print('accuracy = %.4f' % mean_acc)

    elif args.task == 'segmentation':
        mean_dice = np.mean(dice_list)
        print('mean dice = %.4f' % mean_dice)
    else:
        mean_dice = np.mean(dice_list)
        print('mean dice = %.4f' % mean_dice)

        total_target_coords = np.stack(target_coords_list, axis=0)
        total_pred_coords = np.stack(pred_coords_list, axis=0)
        total_seg_coords = np.stack(seg_coords_list, axis=0)
        total_pixelspacings = np.stack(pixelspacing_list, axis=0)
        mean_pred_acc = cal_acc(total_pred_coords, total_target_coords, total_pixelspacings, max_dist=6.0)
        mean_seg_acc = cal_acc(total_seg_coords, total_target_coords, total_pixelspacings, max_dist=6.0)
        print('seg accuracy = %.4f, pred accuracy = %.4f' % (mean_seg_acc, mean_pred_acc))
        total_pred_dists = np.stack(pred_dists_list, axis=0)
        total_seg_dists = np.stack(seg_dists_list, axis=0)
        print('mean seg dist = %.2f mm, pred dist = %.2f mm' % (np.mean(seg_dists_list), np.mean(pred_dists_list)))

    end_time = time.time()
    mean_time = (end_time - start_time) / test_num
    print('task completed, {} seconds used per case'.format(mean_time))