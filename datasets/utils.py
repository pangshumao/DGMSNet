import numpy as np
import time
import torch
from torch.nn import functional as F

def mask2coords(mask, class_num=11):
    '''

    :param mask: A numpy array with shape of (h, w)
    :param class_num:
    :return: A numpy array with shape of (class_num - 1, 2)
    '''
    eps = 1e-8
    coords = np.zeros((class_num - 1, 2), dtype=np.float32)
    for i in range(1, 11):
        pos = np.argwhere(mask == i)
        count = pos.shape[0]
        center_h, center_w = pos.sum(0) / (count + eps)
        center_h = round(center_h)
        center_w = round(center_w)
        coords[i - 1, :] = np.array([center_w, center_h])  # x, y
    return coords

def mask2coords_batch(masks, class_num=11):
    '''

    :param masks: A numpy array with shape of (batch_size, h, w)
    :param class_num:
    :return: A numpy array with shape of (batch_size, class_num - 1, 2)
    '''
    batch_size = masks.shape[0]
    coords = [mask2coords(masks[i, :, :]) for i in range(batch_size)]
    coords = np.stack(coords, axis=0)
    return coords

def pad_image_coords_to_square(image, coords=None):
    '''

    :param image: A numpy array with shape of (h, w)
    :param coords: A numpy array with shape of (point_num, 2), each row denotes the x, y coordinates
    :return:
    '''
    h, w = image.shape
    out = np.zeros((max(h, w), max(h, w)), dtype=image.dtype)
    if h < w:
        pad_h = (w - h) // 2
        out[pad_h : pad_h + h, :] = image
        if coords is not None:
            coords = coords + np.array([[0, pad_h]])
            return out, coords
        else:
            return out
    elif h > w:
        pad_w = (h - w) // 2
        out[:, pad_w : pad_w + w] = image
        if coords is not None:
            coords = coords + np.array([[pad_w, 0]])
            return out, coords
        else:
            return out
    else:
        if coords is None:
            return image
        else:
            return image, coords

def unpad_square_image_coords(square_image, img_size, coords=None):
    '''

    :param square_image: A numpy array with shape of (max(h,w), max(h,w))
    :param img_size: A numpy array, list, or tuple with size of 2, which denotes h and w of the output image
    :param coords: A numpy array with shape of (point_num, 2), each row denotes the x, y coordinates
    :return:
    '''
    h = img_size[0]
    w = img_size[1]
    if h < w:
        pad_h = (w - h) // 2
        out = square_image[pad_h : pad_h + h, :]
        if coords is None:
            return out
        else:
            coords = coords - np.array([[0, pad_h]])
            return out, coords
    elif h > w:
        pad_w = (h - w) // 2
        out = square_image[:, pad_w : pad_w + w]
        if coords is None:
            return out
        else:
            coords = coords - np.array([[pad_w, 0]])
            return out, coords
    else:
        if coords is None:
            return square_image
        else:
            return square_image, coords

def unpad_square_coords(coords, out_img_size):
    '''
    :param coords: A numpy array with shape of (point_num, 2), each row denotes the x, y coordinates
    :param out_img_size: A numpy array, list, or tuple with size of 2, which denotes h and w of the output image
    :return:
    '''
    h = out_img_size[0]
    w = out_img_size[1]
    if h < w:
        pad_h = (w - h) // 2
        coords = coords - np.array([[0, pad_h]])
        return coords
    elif h > w:
        pad_w = (h - w) // 2
        coords = coords - np.array([[pad_w, 0]])
        return coords
    else:
        return coords


def generate_heatmaps(image, spacing, gt_coords, sigma=4.0):
    '''
    generate the heat maps according to the physical distance
    :param image: a numpy array with shape of (h, w)
    :param spacing: a numpy array, i.e., np.array([w_size, h_size])
    :param gt_coords: a numpy array with shape of (point_num, 2)
    :param sigma:
    :return: a numpy array with shape of (point_num, h, w)
    '''
    coord = np.where(image < np.inf)
    # 注意需要反转横纵坐标
    coord = np.stack(coord[::-1], axis=1).reshape(image.shape[0], image.shape[1], 2)

    dist = []
    for point in gt_coords:
        d = (((coord - point) * spacing) ** 2).sum(axis=-1)
        dist.append(np.exp(-d / (2.0 * (sigma ** 2))))
        # dist.append((((coord - point) * spacing) ** 2).sum(dim=-1).sqrt())
    dist = np.stack(dist, axis=0)

    return dist.astype(np.float32)

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW or NxHxW label image to NxCxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW) or 3D input image (NxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW) or 4D output image (NxCxHxW)
    """
    assert input.dim() in [3, 4]

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to 1xNxDxHxW
    # index = input.unsqueeze(0)

    # expand the input tensor to Nx1xDxHxW
    index = input.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_index = index.expand(shape)
        mask = expanded_index == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        index = index.clone()
        index[index == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, index, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, index, 1)

def normalize_coords(coords, h, w):
    '''
    Normalize a batch of coords to range of [-1, 1]
    :param coords: a tensor with shape of (n, class_num - 1, 2)
    :param h: The heigh of the image, integer scalar
    :param w: The width of the image, integer scalar
    :return:
    '''
    n_coords = torch.zeros_like(coords, dtype=torch.float32)
    n_coords[:, :, 0] = coords[:, :, 0] / w * 2 - 1
    n_coords[:, :, 1] = coords[:, :, 1] / h * 2 - 1
    return n_coords

def affine_transform(binary_channel, delta_x, delta_y):
    '''

    :param binary_channel: A binary tensor mask with shape of (n, h, w)
    :param delta_x: a tensor with shape of (n, ), it is calculated by normalized_target_coords_x - normalized_src_coords_x
    :param delta_y: a tensor with shape of (n, ), it is calculated by normalized_target_coords_y - normalized_src_coords_y
    :return:
    '''
    binary_channel = torch.unsqueeze(binary_channel, dim=1) # (n, 1, h, w)
    n = binary_channel.shape[0]
    m = torch.zeros((n, 2, 3), dtype=torch.float32, device=binary_channel.device)
    m[:, 0, 0] = 1.
    m[:, 1, 1] = 1.
    m[:, 0, 2] = -delta_x
    m[:, 1, 2] = -delta_y
    grid = F.affine_grid(m, binary_channel.size(), align_corners=True)
    wraped = F.grid_sample(binary_channel, grid, align_corners=True)
    wraped = wraped.squeeze(dim=1)
    return wraped

def rotate(binary_channel, coords, angle):
    '''
    rotate the L1 or L1/L2 around the center of L1 and L1/L2
    :param binary_channel: A binary tensor mask with shape of (n, h, w)
    :param coords: A tensor with shape of (n, 2)
    :param angle: A tensor with shape of (n, 2)
    :return:
    '''
    angle = angle * np.pi / 180.
    binary_channel = torch.unsqueeze(binary_channel, dim=1)  # (n, 1, h, w)
    n = binary_channel.shape[0]

    '''
    translate to the image center, then rotate
    '''
    m = torch.zeros((n, 2, 3), dtype=torch.float32, device=binary_channel.device)
    delta_x = 0. - coords[:, 0]
    delta_y = 0. - coords[:, 1]
    m[:, 0, 0] = torch.cos(angle)
    m[:, 0, 1] = -torch.sin(angle)
    m[:, 1, 0] = torch.sin(angle)
    m[:, 1, 1] = torch.cos(angle)
    m[:, 0, 2] = -delta_x
    m[:, 1, 2] = -delta_y
    grid = F.affine_grid(m, binary_channel.size(), align_corners=True)
    wraped = F.grid_sample(binary_channel, grid, align_corners=True)

    '''
    translate
    '''
    m = torch.zeros((n, 2, 3), dtype=torch.float32, device=binary_channel.device)
    m[:, 0, 0] = 1
    m[:, 1, 1] = 1
    m[:, 0, 2] = delta_x
    m[:, 1, 2] = delta_y
    grid = F.affine_grid(m, wraped.size(), align_corners=True)
    wraped = F.grid_sample(wraped, grid, align_corners=True)
    wraped = wraped.squeeze(dim=1)
    return wraped


def template_to_moving(template_one_hot, template_coords, moving_coords):
    '''

    :param template_one_hot: a tensor with shape of (n, class_num, h, w)
    :param template_coords: a tensor with shape of (n, class_num - 1, 2)
    :param moving_coords: a tensor with shape of (n, class_num - 1, 2)
    :return:
    '''
    n, class_num, h, w = template_one_hot.shape
    template_one_hot = template_one_hot.float()

    '''
    Normalizing the coords to range of [-1, 1]
    '''
    n_moving_coords = normalize_coords(moving_coords, h, w)
    n_template_coords = normalize_coords(template_coords, h, w)
    template2moving = torch.zeros_like(template_one_hot, dtype=torch.float32)
    for i in range(1, class_num):
        delta_x = n_moving_coords[:, i - 1, 0] - n_template_coords[:, i - 1, 0]
        delta_y = n_moving_coords[:, i - 1, 1] - n_template_coords[:, i - 1, 1]
        template2moving[:, i, :, :] = affine_transform(template_one_hot[:, i, :, :], delta_x, delta_y)

    t2m_sum = torch.sum(template2moving, dim=1)
    for i in range(n):
        template2moving[i, 0, :, :] = torch.where(t2m_sum[i] > 0, torch.full_like(t2m_sum[i], 0), torch.full_like(t2m_sum[i], 1))

    wraped_mask = torch.argmax(template2moving, dim=1)
    wraped_mask_one_hot = expand_as_one_hot(wraped_mask, C=class_num)
    return wraped_mask_one_hot, wraped_mask

def modify_mask_one_hot(mask_one_hot, coords, mode, channel_ind=1):
    '''

    :param mask_one_hot: a tensor with shape of (n, class_num, h, w)
    :param coords: a tensor with shape of (n, class_num - 1, 2)
    :param mode: The modification mode. if mode==0, move L1, L1/L2 to L5, L5/S1 respectively, and
    the other stuctures move down one level correspondingly.
    :param channel_ind: if mode==1, mask_one_hot[:, channel_ind, :, :] will be set to zero
    :return:
    '''
    n, class_num, h, w = mask_one_hot.shape
    mask = torch.argmax(mask_one_hot, dim=1) # (n, h, w)
    n_coords = normalize_coords(coords, h, w)
    if mode == 0:
        temp_mask = mask - 1
        temp_mask = torch.where(temp_mask == 5, torch.full_like(temp_mask, 10), temp_mask) # 5 -> 10
        temp_mask = torch.where(temp_mask == 0, torch.full_like(temp_mask, 5), temp_mask) # 0 -> 5
        temp_mask = torch.where(temp_mask == -1, torch.full_like(temp_mask, 0), temp_mask) # -1 -> 0
        temp_mask_one_hot = expand_as_one_hot(temp_mask, C=class_num)

        angle1 = torch.atan(torch.abs(n_coords[:, 5, 0] - n_coords[:, 0, 0]) / torch.abs(n_coords[:, 5, 1] - n_coords[:, 0, 1])) * 180
        angle2 = torch.atan(torch.abs(n_coords[:, 9, 0] - n_coords[:, 4, 0]) / torch.abs(
            n_coords[:, 9, 1] - n_coords[:, 4, 1])) * 180

        angle = angle1 + angle2
        angle = angle.div(np.pi)
        # angle = np.arctan(torch.abs(n_coords[:, 9, 0] - n_coords[:, 4, 0]) / torch.abs(n_coords[:, 9, 1] - n_coords[:, 4, 1])) * 180 / np.pi * 10

        temp_mask_one_hot[:, 5, :, :] = rotate(temp_mask_one_hot[:, 5, :, :], coords=(n_coords[:, 0, :] + n_coords[:, 5, :]) * 0.5, angle=angle)
        temp_mask_one_hot[:, 10, :, :] = rotate(temp_mask_one_hot[:, 10, :, :], coords=(n_coords[:, 0, :] + n_coords[:, 5, :]) * 0.5, angle=angle)

        delta_x_5 = n_coords[:, 4, 0] - n_coords[:, 0, 0] + torch.abs(n_coords[:, 4, 0] - n_coords[:, 3, 0]) * 2
        delta_y_5 = (n_coords[:, 4, 1] - n_coords[:, 0, 1]) * 1.25
        temp_mask_one_hot[:, 5, :, :] = affine_transform(temp_mask_one_hot[:, 5, :, :], delta_x_5, delta_y_5)

        temp_mask_one_hot[:, 10, :, :] = affine_transform(temp_mask_one_hot[:, 10, :, :], delta_x_5, delta_y_5)

        t2m_sum = torch.sum(temp_mask_one_hot[:, 1:, :, :], dim=1)
        for i in range(n):
            temp_mask_one_hot[i, 0, :, :] = torch.where(t2m_sum[i] > 0, torch.full_like(t2m_sum[i], 0),
                                                        torch.full_like(t2m_sum[i], 1))

        modified_mask = torch.argmax(temp_mask_one_hot, dim=1)
    elif mode == 1:
        mask_one_hot[:, 0, :, :] = torch.where(mask_one_hot[:, channel_ind, :, :] == 1,
                                               mask_one_hot[:, channel_ind, :, :], mask_one_hot[:, 0, :, :])
        mask_one_hot[:, channel_ind, :, :] = torch.full_like(mask_one_hot[:, channel_ind, :, :], 0)
        modified_mask = torch.argmax(mask_one_hot, dim=1)

    modified_mask_one_hot = expand_as_one_hot(modified_mask, C=class_num)
    return modified_mask_one_hot, modified_mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = np.random.rand(880, 880)
    spacing = np.array([0.34, 0.34])
    gt_coords = np.random.randint(0, 127, (10, 2))
    start_time = time.time()
    dist = generate_heatmaps(image, spacing, gt_coords, sigma=3.0)
    plt.imshow(dist[0])
    plt.show()
    end_time = time.time()
    print('time: %f' % (end_time - start_time))


