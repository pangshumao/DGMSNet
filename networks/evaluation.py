import math
import matplotlib.pyplot as plt
import numpy as np


def distance(coord0, coord1, pixel_spacing):
    x = (coord0[0] - coord1[0]) * pixel_spacing[0]
    y = (coord0[1] - coord1[1]) * pixel_spacing[1]
    output = math.sqrt(x ** 2 + y ** 2)
    return output

def cal_distance_per_image(pred_coords, target_coords, pixel_spacings):
    '''
    calculate the distance of the key points for an image, unit: mm
    :param pred_coords: a numpy array with shape of (point_num, 2)
    :param target_coords: a numpy array with shape of (point_num, 2)
    :param pixel_spacings: a numpy array with shape of (2, )
    :return: a numpy array with shape of (point_num, )
    '''
    dists = []
    for i in range(pred_coords.shape[0]):
        dists.append(distance(pred_coords[i], target_coords[i], pixel_spacings))
    dists = np.array(dists, dtype=np.float32)
    return dists

def cal_distances(pred_coords, target_coords, pixel_spacings):
    '''
    calculate the distance of the key points for all images, unit: mm
    :param pred_coords: a numpy array with shape of (sample_num, point_num, 2)
    :param target_coords: a numpy array with shape of (sample_num, point_num, 2)
    :param pixel_spacings: a numpy array with shape of (sample_num, 2)
    :return: a numpy array with shape of (sample_num, point_num)
    '''
    sample_num, point_num = pred_coords.shape[0:2]
    dists = np.zeros((sample_num, point_num), dtype=np.float32)
    for i in range(sample_num):
        dists[i, :] = cal_distance_per_image(pred_coords[i], target_coords[i], pixel_spacings[i])
    return dists

def cal_acc(pred_coords, target_coords, pixel_spacings, max_dist=2.0):
    '''
    calculate the accuracy for key points localization and identification for all images
    :param pred_coords: a numpy array with shape of (sample_num, point_num, 2)
    :param target_coords: a numpy array with shape of (sample_num, point_num, 2)
    :param pixel_spacings: a numpy array with shape of (sample_num, 2)
    :param max_dist: a scalar, unit: mm
    :return:
    '''
    dists = cal_distances(pred_coords, target_coords, pixel_spacings)
    tp = np.where(dists < max_dist, 1, 0).sum()
    count = dists.shape[0] * dists.shape[1]
    acc = tp / count
    return acc

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

def dice_all_class(prediction, target, class_num=11, skip_background=True, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = []
    if skip_background:
        for i in range(1, class_num):
            if i not in target:
                continue
            target_per_class = np.where(target == i, 1, 0)
            prediction_per_class = np.where(prediction == i, 1, 0)
            dice = dice_per_class(prediction_per_class, target_per_class, eps)
            dices.append(dice)
        return np.mean(dices)
    else:
        for i in range(0, class_num):
            if i not in target:
                continue
            target_per_class = np.where(target == i, 1, 0)
            prediction_per_class = np.where(prediction == i, 1, 0)
            dice = dice_per_class(prediction_per_class, target_per_class, eps)
            dices.append(dice)
        return np.mean(dices)

def batch_dice_all_class(predictions, targets, class_num=11, skip_background=True, eps=1e-10):
    '''

    :param predictions: a numpy array with shape of [batch_size, h, w]
    :param targets: a numpy array with shape of [batch_size, h, w]
    :param class_num:
    :param eps:
    :return:
    '''
    batch_size = predictions.shape[0]
    dices = []
    for i in range(batch_size):
        dices.append(dice_all_class(predictions[i], targets[i], class_num, skip_background, eps))
    return np.mean(dices)



