import math
import os
import pickle
import time
import traceback
from copy import deepcopy
from typing import List
from tensorboardX import SummaryWriter
import torch
from networks.evaluation import cal_acc
import numpy as np

from .utils import tqdm

class Trainer:
    def __init__(self, module: torch.nn.Module, train_data, val_data, optimizer, epochs, loss, is_higher_better,
        batch_num, early_stopping=-1, scheduler=None, max_dist=2.0,
        checkpoint_dir=None):
        self.module = module
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = loss
        self.is_higher_better = is_higher_better
        self.batch_num = batch_num
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.max_dist = max_dist
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
            'accuracy',
            'higher' if self.is_higher_better else 'lower'))

        best_state_dict = deepcopy(self.module.state_dict())
        best_step = -1
        best_metric_value = None
        lowest_val_loss = 1000000.0
        loss_record = []
        generator = self.forever_iter()
        try:
            for epoch in range(self.epochs):
                # time.sleep(0.5)
                self.module.train(True)
                pred_coords_list = []
                target_coords_list = []
                pixelspacings_list = []

                for _ in tqdm(range(self.batch_num), ascii=True):
                    # --------- 训练参数 ------------
                    images, target_heatmaps, target_coords, pixelspacings = next(generator)

                    self.step += 1
                    images = images.cuda()
                    target_heatmaps = target_heatmaps.cuda()

                    self.optimizer.zero_grad()
                    pred_coords, pred_heatmaps = self.module(images, return_more=True)

                    loss_value = self.loss(pred_heatmaps, target_heatmaps)

                    loss_record.append(float(loss_value.detach()))
                    pred_coords_list.append(pred_coords.detach().to('cpu').numpy())
                    target_coords_list.append(target_coords.detach().to('cpu').numpy())
                    pixelspacings_list.append(pixelspacings.detach().to('cpu').numpy())

                    if loss_value is not None:
                        loss_value.backward()
                        self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.epoch += 1
                if self.checkpoint_dir is not None:
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'last.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'last.weight'))

                pred_coords_np = np.concatenate(pred_coords_list, axis=0)
                target_coords_np = np.concatenate(target_coords_list, axis=0)
                pixelspacings_np = np.concatenate(pixelspacings_list, axis=0)

                train_metric_value = cal_acc(pred_coords_np, target_coords_np, pixelspacings_np, max_dist=self.max_dist)

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
                    best_state_dict = deepcopy(self.module.state_dict())
                    best_step = self.step
                    best_metric_value = val_metric_value
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'best.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'best.weight'))

                    with torch.no_grad():
                        print('epoch {} train accuracy: {} train loss: {}'.format(epoch, train_metric_value, train_loss_avg))
                    print('valid best accuracy: {} loss: {}....................................................'
                          .format(val_metric_value, val_loss))
                else:
                    with torch.no_grad():
                        print('epoch {} train accuracy: {} train loss: {}'.format(epoch, train_metric_value, train_loss_avg))
                    print('valid accuracy: {} loss: {}'.format(val_metric_value, val_loss))
                # ------ 提前停止的策略
                if self.step - best_step >= self.early_stopping > 0:
                    break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        finally:
            self.module.load_state_dict(best_state_dict)
        return best_metric_value, loss_record


    def validate(self):
        self.module.eval()
        loss_list = []
        pred_coords_list = []
        target_coords_list = []
        pixelspacings_list = []
        for i, t in enumerate(self.val_data):
            images, target_heatmaps, target_coords, pixelspacings = t
            images = images.cuda()
            target_heatmaps = target_heatmaps.cuda()

            pred_coords, pred_heatmaps = self.module(images, return_more=True)

            loss_value = self.loss(pred_heatmaps, target_heatmaps)

            loss_list.append(float(loss_value.detach()))
            pred_coords_list.append(pred_coords.detach().to('cpu').numpy())
            target_coords_list.append(target_coords.detach().to('cpu').numpy())
            pixelspacings_list.append(pixelspacings.detach().to('cpu').numpy())

        pred_coords_np = np.concatenate(pred_coords_list, axis=0)
        target_coords_np = np.concatenate(target_coords_list, axis=0)
        pixelspacings_np = np.concatenate(pixelspacings_list, axis=0)
        metric_value = cal_acc(pred_coords_np, target_coords_np, pixelspacings_np, max_dist=self.max_dist)
        return np.mean(loss_list), metric_value
