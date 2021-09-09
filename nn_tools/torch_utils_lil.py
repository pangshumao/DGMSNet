import os
from tensorboardX import SummaryWriter
import torch
from networks.evaluation import cal_acc, batch_dice_all_class
import numpy as np
from torch.nn import functional as F
from .utils import tqdm

class Trainer:
    def __init__(self, module: torch.nn.Module, train_data, val_data, optimizer, epochs, l_loss, p_loss, beta,
                 is_higher_better,
        batch_num, early_stopping=-1, scheduler=None,
        checkpoint_dir=None, max_dist=3.0, devices=None, labeled_batch_size=1):
        self.module = module
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.epochs = epochs
        self.l_loss = l_loss
        self.p_loss = p_loss
        self.beta = beta
        self.is_higher_better = is_higher_better
        self.batch_num = batch_num
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.max_dist = max_dist
        self.devices = devices
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.step = 0
        self.epoch = 0
        self.labeled_batch_size = labeled_batch_size

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
        best_step = -1
        best_metric_value = None
        lowest_val_loss = 1000000.0
        p_loss_record = []
        l_loss_record = []
        loss_record = []
        generator = self.forever_iter()
        try:
            for epoch in range(self.epochs):
                # time.sleep(0.5)
                self.module.train(True)
                train_dices = []
                pred_coords_list = []
                target_coords_list = []
                pixelspacings_list = []

                for _ in tqdm(range(self.batch_num), ascii=True):
                    # --------- 训练参数 ------------
                    images, target_masks, target_heatmaps, target_coords, pixelspacings = next(generator)

                    self.step += 1
                    images = images.to(self.devices[0])
                    target_masks = target_masks.to(self.devices[0])
                    target_heatmaps = target_heatmaps.to(self.devices[0])

                    self.optimizer.zero_grad()
                    logits, pred_heatmaps, pred_coords = self.module(images, return_more=True)

                    p_loss_value = self.p_loss(logits[:self.labeled_batch_size], target_masks[:self.labeled_batch_size])

                    l_loss_value = self.l_loss(pred_heatmaps, target_heatmaps)

                    loss_value = p_loss_value + self.beta * l_loss_value

                    prob = F.softmax(logits[:self.labeled_batch_size], dim=1)
                    seg = torch.argmax(prob, dim=1).detach().to('cpu').numpy() # (n, h, w)
                    target_masks = target_masks[:self.labeled_batch_size].detach().to('cpu').numpy()  # (n, h, w)
                    dice = batch_dice_all_class(seg, target_masks, class_num=logits.shape[1])
                    train_dices.append(dice)

                    p_loss_record.append(float(p_loss_value.detach()))
                    l_loss_record.append(float(l_loss_value.detach()))
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

                train_dice_value = np.mean(train_dices)
                pred_coords_np = np.concatenate(pred_coords_list, axis=0)
                target_coords_np = np.concatenate(target_coords_list, axis=0)
                pixelspacings_np = np.concatenate(pixelspacings_list, axis=0)

                train_acc_value = cal_acc(pred_coords_np, target_coords_np, pixelspacings_np, max_dist=self.max_dist)

                val_loss, val_p_loss, val_l_loss, val_dice_value, val_acc_value = self.validate()

                train_loss_avg = torch.tensor(
                                [x for x in loss_record[-self.batch_num:] if x is not None]).mean()
                train_l_loss_avg = torch.tensor(
                    [x for x in l_loss_record[-self.batch_num:] if x is not None]).mean()
                train_p_loss_avg = torch.tensor(
                    [x for x in p_loss_record[-self.batch_num:] if x is not None]).mean()


                self._log_train_val_stats('loss', train_loss_avg, val_loss)
                self._log_train_val_stats('localization_loss', train_l_loss_avg, val_l_loss)
                self._log_train_val_stats('parsing_loss', train_p_loss_avg, val_p_loss)

                self._log_train_val_stats('acc', train_acc_value, val_acc_value)
                self._log_train_val_stats('dice', train_dice_value, val_dice_value)

                # self._log_params()
                self._log_lr()

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'lowest_loss.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'lowest_loss.weight'))

                if (best_metric_value is None
                        or val_dice_value == best_metric_value
                        or self.is_higher_better == (val_dice_value > best_metric_value)):
                    best_step = self.step
                    best_metric_value = val_dice_value
                    torch.save(self.module, os.path.join(self.checkpoint_dir, 'best.module'))
                    torch.save(self.module.state_dict(), os.path.join(self.checkpoint_dir, 'best.weight'))

                    with torch.no_grad():
                        print('epoch {} train dice: {} acc: {} loss: {} p_loss: {} l_loss: {}'
                              ''.format(epoch, train_dice_value, train_acc_value, train_loss_avg, train_p_loss_avg,
                                        train_l_loss_avg))
                        print('valid best dice: {} acc: {} loss: {} p_loss: {} l_loss: {}....................................................'
                          .format(val_dice_value, val_acc_value, val_loss, val_p_loss, val_l_loss))
                else:
                    with torch.no_grad():
                        print('epoch {} train dice: {} acc: {} loss: {} p_loss: {} l_loss: {}'
                              ''.format(epoch, train_dice_value, train_acc_value, train_loss_avg, train_p_loss_avg,
                                        train_l_loss_avg))
                        print(
                            'valid dice: {} acc: {} loss: {} p_loss: {} l_loss: {}'
                            .format(val_dice_value, val_acc_value, val_loss, val_p_loss, val_l_loss))
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
        l_loss_list = []
        p_loss_list = []
        dices = []
        pred_coords_list = []
        target_coords_list = []
        pixelspacings_list = []
        for i, t in enumerate(self.val_data):
            images, target_masks, target_heatmaps, target_coords, pixelspacings = t
            images = images.to(self.devices[0])
            target_masks = target_masks.to(self.devices[0])
            target_heatmaps = target_heatmaps.to(self.devices[0])

            logits, pred_heatmaps, pred_coords = self.module(images, return_more=True)

            p_loss_value = self.p_loss(logits, target_masks)
            l_loss_value = self.l_loss(pred_heatmaps, target_heatmaps)
            loss_value = p_loss_value + self.beta * l_loss_value

            l_loss_list.append(float(l_loss_value.detach()))
            p_loss_list.append(float(p_loss_value.detach()))
            loss_list.append(float(loss_value.detach()))
            pred_coords_list.append(pred_coords.detach().to('cpu').numpy())
            target_coords_list.append(target_coords.detach().to('cpu').numpy())
            pixelspacings_list.append(pixelspacings.detach().to('cpu').numpy())

            prob = F.softmax(logits, dim=1)
            seg = torch.argmax(prob, dim=1).detach().to('cpu').numpy()  # (n, h, w)
            target_masks = target_masks.detach().to('cpu').numpy()  # (n, h, w)
            dice = batch_dice_all_class(seg, target_masks, class_num=logits.shape[1])

            dices.append(dice)

        dice_value = np.mean(dices)
        pred_coords_np = np.concatenate(pred_coords_list, axis=0)
        target_coords_np = np.concatenate(target_coords_list, axis=0)
        pixelspacings_np = np.concatenate(pixelspacings_list, axis=0)
        acc_value = cal_acc(pred_coords_np, target_coords_np, pixelspacings_np, max_dist=self.max_dist)
        return np.mean(loss_list), np.mean(p_loss_list), np.mean(l_loss_list), dice_value, acc_value
