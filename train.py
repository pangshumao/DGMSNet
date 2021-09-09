import sys
import time
import os
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from datasets.h5_weak_label import load_data
from networks.deeplabv3_plus import DeepLabv3_plus_2d as DeepLabv3_plus
sys.path.append('nn_tools/')
from nn_tools import torch_utils_keypoint, torch_utils_seg, torch_utils_lil
from nn_tools import torch_utils_gcn_seg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nn_tools.utils import get_number_of_learnable_parameters, load_checkpoint
from networks.deeplabv3_plus_lil import DeepLabv3_plus_lil
from networks.deeplabv3_plus_gcn import DeepLabv3_plus_gcn
import numpy as np
import random

import shutil

def init_seeds(seed=0):
    version = float(torch.__version__.split('+')[0][:-2])
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if version > 1.7:
    #     torch.use_deterministic_algorithms(True)
    # else:
    #     torch.set_deterministic(True)

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)


    parser.add_argument("--model", type=str, default='DeepLabv3_plus_lil',
                        help="the model name, "
                             "DeepLabv3_plus"
                             "DeepLabv3_plus_gcn, "
                             "DeepLabv3_plus_lil")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/DGMSNet/Spinal_disease_dataset',
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

    parser.add_argument('--batch_size', type=int, default=2,
                        help="The batch size")

    parser.add_argument("--use_weak_label", dest='use_weak_label', action='store_true',
                        help="use use_weak_label")

    parser.add_argument("--no-use_weak_label", dest='use_weak_label', action='store_false',
                        help="without using use_weak_label")

    parser.set_defaults(use_weak_label=False)

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, MSELoss, CrossEntropyLoss.")

    parser.add_argument('--beta', type=float, default=100.0,
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

    parser.add_argument("--epochs", type=int, default=100,
                        help="max number of epochs")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam, AdamW, or SGD")

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='the weight decay')

    parser.add_argument('--learning_rate', type=float, default=3e-3,
                        help="The initial learning rate")

    parser.add_argument("--seed", type=int, default=0,
                        help="init seed")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    init_seeds(seed=args.seed)

    data_dir = args.data_dir
    fold_ind = args.fold_ind
    file_path = os.path.join(os.path.join(data_dir, 'in', 'h5py', 'data_fold' + str(fold_ind) + '.h5'))
    fold_dir = os.path.join(data_dir, 'model', 'fold' + str(fold_ind))
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    batch_size = args.batch_size

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
        model = DeepLabv3_plus_lil(in_channels=1, l_out_channels=10, p_out_channels=11, devices=args.devices,
                                       os=args.os)
        if args.batch_size > 2:
            identifier = identifier + '_batch_size_' + str(args.batch_size)
    else:
        if args.model == 'DeepLabv3_plus':
            if len(args.devices) == 1:
                gpu_index = args.devices[0].split(':')[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            model = DeepLabv3_plus(out_channels=out_channels, os=args.os)
            model.cuda()
        elif args.model == 'DeepLabv3_plus_gcn':
            if len(args.devices) == 1:
                gpu_index = args.devices[0].split(':')[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            model = DeepLabv3_plus_gcn(out_channels=out_channels, os=args.os)
            if not args.pre_trained:
                model.cuda()

    if args.epochs != 100:
        identifier = identifier + '_epochs_' + str(args.epochs)

    if args.weight_decay != 1e-4:
        identifier = identifier + '_weight_decay_' + str(args.weight_decay)

    print('Number of learnable parameters: %d' % get_number_of_learnable_parameters(model))
    print(model)

    if args.use_weak_label:
        labeled_batch_size = batch_size - batch_size // 2
    else:
        labeled_batch_size = batch_size

    training_data_loader, val_data_loader, testing_data_loader, f = load_data(file_path, prob_rotate=0.4,
        max_angle=15, batch_size=batch_size, labeled_bs=labeled_batch_size, num_workers=0, task=args.task,
                                                                          use_weak_label=args.use_weak_label,
                                                                          use_histograms_match=False)

    batch_num = len(training_data_loader)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 3, args.epochs // 1.5],
                                                         gamma=0.2)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 3, args.epochs // 1.5],
                                                         gamma=0.2)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    if args.loss == 'MSELoss':
        loss = MSELoss()
    elif args.loss == 'CrossEntropyLoss':
        loss = CrossEntropyLoss()

    if not args.use_scheduler:
        identifier = identifier + '_no-use_scheduler'
        scheduler = None

    if args.model == 'DeepLabv3_plus_gcn':
        identifier = identifier + '_beta_' + str(args.beta)
        if args.gcn_num != 3:
            identifier += '_gcn_num_' + str(args.gcn_num)
        if args.pre_trained:
            identifier += '_pre_trained'
            pre_trained_model_path = os.path.join(fold_dir, args.pre_trained_identifier, 'best.weight')
            load_checkpoint(pre_trained_model_path, model)
            model = model.cuda()

    if args.seed != 0:
        identifier += '_seed_' + str(args.seed)


    model_dir = os.path.join(fold_dir, identifier)


    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    else:
        os.makedirs(model_dir)

    if args.task == 'localization':
        trainer = torch_utils_keypoint.Trainer(
            model,
            train_data=training_data_loader,
            val_data=val_data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            loss=loss,
            is_higher_better=True,
            batch_num=batch_num,
            max_dist=3.0,
            checkpoint_dir=model_dir
        )
    elif args.task == 'segmentation':
        if args.model == 'DeepLabv3_plus_gcn':
            trainer = torch_utils_gcn_seg.Trainer(
                model,
                train_data=training_data_loader,
                val_data=val_data_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.epochs,
                loss=loss,
                is_higher_better=True,
                batch_num=batch_num,
                beta=args.beta,
                checkpoint_dir=model_dir
            )
        else:
            trainer = torch_utils_seg.Trainer(
                model,
                train_data=training_data_loader,
                val_data=val_data_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.epochs,
                loss=loss,
                is_higher_better=True,
                batch_num=batch_num,
                checkpoint_dir=model_dir
            )
    else:
        trainer = torch_utils_lil.Trainer(
            model,
            train_data=training_data_loader,
            val_data=val_data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            l_loss=MSELoss(),
            p_loss=CrossEntropyLoss(),
            beta=args.beta,
            is_higher_better=True,
            batch_num=batch_num,
            checkpoint_dir=model_dir,
            max_dist=3.0,
            devices=args.devices,
            labeled_batch_size=labeled_batch_size
        )
    trainer.train()
    f.close()

    print('task completed, {} seconds used'.format(time.time() - start_time))
