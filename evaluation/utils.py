import scipy.sparse as sparse
import numpy as np
import torch
from medpy.metric.binary import assd, dc



class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def dice_per_class(prediction, target, eps=1e-10):
    '''

    :param prediction: numpy array
    :param target: numpy array
    :param eps:
    :return:
    '''
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return (2. * intersect / (np.sum(prediction) + np.sum(target) + eps))

def intersect_per_class(prediction, target, eps=1e-10):
    '''

    :param prediction: numpy array
    :param target: numpy array
    :param eps:
    :return:
    '''
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return intersect, np.sum(prediction), np.sum(target)

def dice_all_class(prediction, target, class_num=11, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = []
    for i in range(1, class_num):
        if i not in target:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        dice = dice_per_class(prediction_per_class, target_per_class)
        dices.append(dice)
    return np.mean(dices)

def dices_each_class(prediction, target, class_num=11, eps=1e-10, empty_value=-1.0):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        # dice = dice_per_class(prediction_per_class, target_per_class)
        dice = dc(prediction_per_class, target_per_class)
        dices[i] = dice
    return dices

def dice_whole_class(prediction, target, class_num=11, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    intersect_sum = 0
    prediction_sum = 0
    target_sum = 0
    for i in range(1, class_num):
        if i not in target:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        result = intersect_per_class(prediction_per_class, target_per_class)
        intersect_sum += result[0]
        prediction_sum += result[1]
        target_sum += result[2]
    return (2. * intersect_sum / (prediction_sum + target_sum + eps))

def assds_each_class(prediction, target, class_num=11, eps=1e-10, voxel_size=(1,1,1), empty_value=-1.0, connectivity=1):
    assds = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target:
            continue
        if i not in prediction:
            print('label %d is zero' % i)
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        ad = assd(prediction_per_class, target_per_class, voxelspacing=voxel_size, connectivity=connectivity)
        assds[i] = ad
    return assds

def evaluation_metrics_each_class(prediction, target, class_num=11, eps=1e-10, empty_value=-1.0):
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    precisions = empty_value * np.ones((class_num), dtype=np.float32)
    recalls = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dsc = 2 * tp / (2 * tp + fp + fn + eps)

        dscs[i] = dsc
        precisions[i] = precision
        recalls[i] = recall
    return dscs, precisions, recalls

def evaluation_accuracy(prediction, target, class_num=11):
    voxel_num = np.size(target)
    voxel_num = np.float32(voxel_num)
    tp = 0.0
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp += np.sum(prediction_per_class * target_per_class)

    accuracy = tp / voxel_num
    return accuracy

def np_onehot(label, num_classes):
    return np.eye(num_classes)[label.astype(np.int32)]
