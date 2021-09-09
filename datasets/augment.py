import numpy as np
from skimage import transform
import math
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import os
from datasets.utils import generate_heatmaps

def resize_point(points, in_img_size, out_img_size):
    '''
    resize the coordinates of the key points
    :param points: a numpy array with shape of (point_num, 2)
    :param in_img_size: a tuple of list which denotes [in_h, in_w]
    :param out_img_size: a tuple of list which denotes [out_h, out_w]
    :return:
    '''
    height_ratio = out_img_size[0] / in_img_size[0]
    width_ratio = out_img_size[1] / in_img_size[1]

    ratio = np.array([width_ratio, height_ratio])
    points = points * ratio
    return points

def rotate_point(points, angel, center):
    '''
    rotate the coordinates of the key points
    :param points: a numpy array with shape of (point_num, 2)
    :param angel: the rotation degree
    :param center: a tuple or list which denotes [center_w, center_h]
    :return:
    '''
    if angel == 0:
        return points
    angel = angel * math.pi / 180
    if isinstance(center, list) or isinstance(center, tuple):
        center = np.array(center)
    while len(center.shape) < len(points[0].shape):
        center = center.unsqueeze(0)
    cos = math.cos(angel)
    sin = math.sin(angel)
    rotate_mat = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    output = points - center
    output = np.matmul(output, rotate_mat)
    return output + center

def rotate_image(image, angle, interpolation, center=None):
    '''
    rotate the 2D image
    :param image: a numpy array with shape of (h, w) or (c, h, w)
    :param angle: the rotation degree
    :param interpolation: 'bilinear', 'cubic', or 'nearest'
    :param center: a tuple which denotes (center_w, center_h), if None, we use the center of the image as the rotation center.
    :return: a numpy array with shape of (h, w) or (c, h, w)
    '''
    if interpolation == 'bilinear':
        interpolation_mode = cv2.INTER_LINEAR
    elif interpolation == 'cubic':
        interpolation_mode = cv2.INTER_CUBIC
    elif interpolation == 'nearest':
        interpolation_mode = cv2.INTER_NEAREST
    def rotate_2d_image(image_2d, angle, interpolation_mode, center):
        h, w = image_2d.shape
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(image_2d.astype(np.float32), M, (w, h), flags=interpolation_mode, borderMode=cv2.BORDER_REFLECT)
        return rotated.astype(image_2d.dtype)
    if image.ndim == 2:
        return rotate_2d_image(image, angle, interpolation_mode, center)
    elif image.ndim == 3:
        rotated = [rotate_2d_image(image[i], angle, interpolation_mode, center)for i in range(image.shape[0])]
        rotated = np.stack(rotated, axis=0)
        return rotated

def rotate(image, points, angle, interpolation='bilinear'):
    '''
    rotate the 2D image and the coordinates of the keypoints
    :param image: a numpy array with shape of (h, w)
    :param points: a numpy array with shape of (point_num, 2)
    :param angle: the rotation degree
    :param interpolation: 'bilinear', 'cubic', or 'nearest'
    :return:
    '''
    h, w = image.shape
    center = (w / 2, h /2)
    return rotate_image(image, angle, interpolation, center), rotate_point(points, angle, center)

if __name__ == '__main__':
    mr_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/MR'
    mask_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/nii/processed_mask'
    kp_dir = '/public/pangshumao/data/Spine_Localization_PIL/in/keypoints'

    img_size = (880, 880)

    for case_ind in range(1, 216):
        nii = nib.load(os.path.join(mr_dir, 'Case' + str(case_ind) + '.nii.gz'))
        mr = nii.get_data()
        mr = mr[:, :, 0]
        mr = np.rot90(mr)
        in_img_size = mr.shape
        hdr = nii.header
        pixel_size_h, pixel_size_w = hdr['pixdim'][1:3]

        height, width = mr.shape
        pixel_size_h = pixel_size_h * height / img_size[0]
        pixel_size_w = pixel_size_w * width / img_size[1]

        nii = nib.load(os.path.join(mask_dir, 'Case' + str(case_ind) + '.nii.gz'))
        mask = nii.get_data()
        mask = mask[:, :, 0]
        mask = np.rot90(mask)

        kp_data = np.load(os.path.join(kp_dir, 'Case' + str(case_ind) + '.npz'))
        points = kp_data['coords']

        mr = transform.resize(mr.astype(np.float32), img_size, order=1, mode='constant', anti_aliasing=False)
        mask = transform.resize(mask.astype(np.float32), img_size, order=0, mode='constant', anti_aliasing=False).astype(
            np.int)
        points = resize_point(points, in_img_size=in_img_size, out_img_size=img_size)
        heatmaps = generate_heatmaps(mr, spacing=np.array([pixel_size_w, pixel_size_h]), gt_coords=points)

        angle = 15
        mr, points = rotate(mr, points, angle=angle, interpolation='bilinear')
        mask = rotate_image(mask, angle=angle, interpolation='nearest')
        heatmaps = rotate_image(heatmaps, angle=angle, interpolation='bilinear')

        mean = np.mean(mr)
        std = np.std(mr)

        mr = mr - mean
        mr = mr / std

        heatmaps_sum = np.sum(heatmaps, axis=0)

        plt.subplot(121)
        plt.imshow(mr, cmap='gray')
        point_num = points.shape[0]
        for i in range(point_num):
            plt.scatter(x=points[i, 0], y=points[i, 1], c='r')
        plt.title('Case%d' % case_ind)
        plt.subplot(122)
        plt.imshow(heatmaps_sum)
        plt.title('Case%d' % case_ind)
        plt.show()

