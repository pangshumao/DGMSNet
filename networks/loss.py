import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

class NullLoss:
    def __call__(self, x, *args):
        return x.mean()

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
        result = torch.zeros(shape, device=input.device).scatter_(1, index, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape, device=input.device).scatter_(1, index, 1)

class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        '''

        :param input: tensor with shape of [N, C, D, H, W]
        :param target: tensor with shape of [N, D, H, W]
        :param weights: tensor with shape of [N, D, H, W]
        :return:
        '''
        assert target.size() == weights.size()

        # normalize the input
        log_probabilities = self.log_softmax(input)

        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)

        # expand weights
        # denominator = weights.sum()
        weights = weights.unsqueeze(1) # [N, 1, H, W]
        weights = weights.expand_as(input) # [N, C, H, W]

        # mask ignore_index if present

        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask

        # compute the losses
        result = -weights * target * log_probabilities

        # temp = result.to('cpu').numpy()
        # plt.subplot(131)
        # plt.imshow(temp[0, 0, :, :])
        #
        # plt.subplot(132)
        # plt.imshow(temp[0, 1, :, :])
        #
        # plt.subplot(133)
        # plt.imshow(temp[0, 2, :, :])
        #
        # plt.show()

        # average the losses
        # loss = torch.div(result.sum(), denominator)
        loss = result.mean(dim=(0, 2, 3)).sum()

        return loss

class MSEDSLoss:
    def __init__(self, max_steps):
        self.step = 0
        self.max_steps = max_steps
        self.loss = torch.nn.MSELoss()
    def __call__(self, pred, target):
        previous_pred = pred[0]
        final_pred = pred[1]
        if self.step < self.max_steps * 0.25:
            lamda = 0.5
        elif self.step < self.max_steps * 0.5:
            lamda = 0.3
        elif self.step < self.max_steps * 0.75:
            lamda = 0.1
        else:
            lamda = 0.0
        loss = lamda * self.loss(previous_pred, target) + (1 - lamda) * self.loss(final_pred, target)
        self.step = self.step + 1
        return loss

class KeyPointBCELoss:
    def __init__(self, max_dist=8):
        self.max_dist = max_dist

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        dist = dist.to(pred.device)

        pred = pred[mask]
        dist = dist[mask]
        label = dist < self.max_dist
        label = label.to(pred.dtype)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1 / label.mean())
        return loss(pred, label)

if __name__ == '__main__':
    loss_fun = PixelWiseCrossEntropyLoss()
    logit = torch.rand(2, 11, 780, 780)
    target = torch.randint(0, 11, size=(2, 780, 780))
    weight = torch.rand(2, 780, 780)
    out = loss_fun(logit, target, weight)