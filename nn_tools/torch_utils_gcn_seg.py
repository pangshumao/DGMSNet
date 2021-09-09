import math
import os
import pickle
import time
import traceback
from copy import deepcopy
from typing import List
from tensorboardX import SummaryWriter
import torch
from networks.evaluation import cal_acc, batch_dice_all_class
import numpy as np
from torch.nn import functional as F

from .utils import tqdm

class Trainer:
    def __init__(self, module: torch.nn.Module, train_data, val_data, optimizer, epochs, loss, is_higher_better,
        batch_num, beta=0.3, early_stopping=-1, scheduler=None,
        checkpoint_dir=None):
        self.module = module
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = loss
        self.is_higher_better = is_higher_better
        self.batch_num = batch_num
        self.beta = beta
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.step = 0
        self.epoch = 0

    def forever_iter(self):
        while True:
            for _ in self.train_data:
                yield _

    def _log_stats(self, phase, loss_avg=None, eval_score_avg=None):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            if value is not None:
                self.writer.add_scalar(tag, value, self.epoch)

    def _log_train_val_stats(self, main_tag, train_value, val_value):
        self.writer.add_scalars(main_tag, {'train': train_value, 'val': val_value}, self.epoch)

    def _log_params(self):
        for name, value in self.module.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.epoch)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.epoch)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        print('learning_rate = %f' % lr)
        self.writer.add_scalar('learning_rate', lr, self.epoch)

    def train(self):
        # 状态变量
        print('using {} as training loss, using {}({} is better) as early stopping metric'.format(
            type(self.loss).__name__,
            'dice',
            'higher' if self.is_higher_better else 'lower'))

        best_step = -1
        best_metric_value = None
        lowest_val_loss = 1000000.0
        loss_record = []
        generator = self.forever_iter()
        try:
            for epoch in range(self.epochs):
                # time.sleep(0.5)
                self.module.train(True)
                train_dices = []

                for _ in tqdm(range(self.batch_num), ascii=True):
                    # --------- 训练参数 ------------
                    images, target_masks, target_coords, pixelspacings = next(generator)

                    self.step += 1
                    images = images.cuda()
                    target_masks = target_masks.cuda()

                    self.optimizer.zero_grad()
                    logit, fea_logit = self.module(images)
                    up_fea_logit = F.interpolate(fea_logit, size=target_masks.size()[1:], mode='bilinear',
                                                 align_corners=True)

                    loss_value = self.loss(logit, target_masks) + self.beta * self.loss(up_fea_logit, target_masks)
                    prob = F.softmax(logit, dim=1)
                    seg = torch.argmax(prob, dim=1).detach().to('cpu').numpy() # (n, h, w)
                    target_masks = target_masks.detach().to('cpu').numpy()  # (n, h, w)
                    dice = batch_dice_all_class(seg, target_masks, class_num=logit.shape[1])
                    train_dices.append(dice)

                    loss_record.append(float(loss_value.detach()))

                    if loss_value is not None:
                        loss_value.backward()
                        self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.epoch += 1
                if self.checkpoint_dir is not None:
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'last.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'last.weight'))

                train_metric_value = np.mean(train_dices)

                val_loss, val_metric_value = self.validate()
                if np.isnan(val_loss):
                    print('val_loss is nan')
                if np.isnan(val_metric_value):
                    print('val_metric_value is nan')

                train_loss_avg = torch.tensor(
                                [x for x in loss_record[-self.batch_num:] if x is not None]).mean()

                if np.isnan(train_loss_avg):
                    print('train_loss_avg is nan')
                if np.isnan(train_metric_value):
                    print('train_metric_value is nan')

                self._log_train_val_stats('loss', train_loss_avg, val_loss)

                self._log_train_val_stats('acc', train_metric_value, val_metric_value)

                # self._log_params()
                self._log_lr()

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'lowest_loss.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'lowest_loss.weight'))

                if (best_metric_value is None
                        or val_metric_value == best_metric_value
                        or self.is_higher_better == (val_metric_value > best_metric_value)):
                    # best_state_dict = deepcopy(self.module.state_dict())
                    best_step = self.step
                    best_metric_value = val_metric_value
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'best.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'best.weight'))

                    with torch.no_grad():
                        print('epoch {} train dice: {} train loss: {}'.format(epoch, train_metric_value, train_loss_avg))
                    print('valid best dice: {} loss: {}....................................................'
                          .format(val_metric_value, val_loss))
                else:
                    with torch.no_grad():
                        print('epoch {} train dice: {} train loss: {}'.format(epoch, train_metric_value, train_loss_avg))
                    print('valid dice: {} loss: {}'.format(val_metric_value, val_loss))
                # ------ 提前停止的策略
                if self.step - best_step >= self.early_stopping > 0:
                    break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        # finally:
        #     self.module.load_state_dict(best_state_dict)
        return best_metric_value, loss_record


    def validate(self):
        self.module.eval()
        loss_list = []
        dices = []
        for i, t in enumerate(self.val_data):
            images, target_masks, target_coords, pixelspacings = t
            images = images.cuda()
            target_masks = target_masks.cuda()

            logit, fea_logit = self.module(images)
            up_fea_logit = F.interpolate(fea_logit, size=target_masks.size()[1:], mode='bilinear',
                                         align_corners=True)

            loss_value = self.loss(logit, target_masks) + self.beta * self.loss(up_fea_logit, target_masks)
            prob = F.softmax(logit, dim=1)
            seg = torch.argmax(prob, dim=1).detach().to('cpu').numpy()  # (n, h, w)
            target_masks = target_masks.detach().to('cpu').numpy()  # (n, h, w)
            dice = batch_dice_all_class(seg, target_masks, class_num=logit.shape[1])

            loss_list.append(float(loss_value.detach()))
            dices.append(dice)

        metric_value = np.mean(dices)
        return np.mean(loss_list), metric_value
