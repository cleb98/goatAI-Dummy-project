from time import time

import numpy as np
import torch
import torchvision as tv
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import Conf
from dataset.dummy_ds import DummyDS
from models import DummyModel
from post_processing import PostProcessor
from progress_bar import ProgressBar
from scheduler import LRScheduler


def get_batch_iou(masks_pred, masks_true):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Compute the intersection over union between two batches of binary masks.

    :param masks_pred: predicted binary masks
        ->> shape: (B, 1, H, W); values: bin{0, 1}
    :param masks_true: target binary masks
        ->> shape: (B, 1, H, W); values: bin{0, 1}
    :return: IoU value for each element in the batch
        ->> shape: (B,); values: range[0, 1]
    """
    inters = torch.logical_and(masks_pred, masks_true).sum((1, 2, 3))
    union = torch.logical_or(masks_pred, masks_true).sum((1, 2, 3))
    return torch.where(union != 0, inters / union, union)


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        """
        :param cnf: configuration object that contains all the
            hyper-parameters and paths.
        """

        self.cnf = cnf

        # init model
        self.model = DummyModel()
        self.model = self.model.to(cnf.device)

        # init optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(), lr=cnf.lr
        )

        # init train loader
        training_set = DummyDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True,
            worker_init_fn=training_set.wif,
        )

        # init val loader
        val_set = DummyDS(cnf, mode='val')
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=cnf.batch_size,
            num_workers=1, shuffle=False, pin_memory=True,
            worker_init_fn=val_set.wif_val,
        )

        self.post_proc = PostProcessor(out_ch_order='RGB')

        # init learning rate scheduler
        self.scheduler = LRScheduler(
            optimizer=self.optimizer,
            max_lr=cnf.lr, final_lr=(cnf.lr / 100),
            epochs=cnf.epochs, dl_len=len(self.train_loader),
            warmup=int(round(cnf.epochs * 0.03))
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.absolute()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values
        self.epoch = 0
        self.best_val_loss = None
        self.patience = self.cnf.max_patience

        # init progress bar
        self.progress_bar = ProgressBar(
            max_step=self.log_freq, max_epoch=self.cnf.epochs
        )

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        Load training checkpoint from the log directory.
        If no checkpoint is found, nothing happens.
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, weights_only=False)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.scheduler.load_state_dict(ck['scheduler'])
            self.scheduler.optimizer = self.optimizer
            self.best_val_loss = ck['best_val_loss']
            self.patience = ck['patience']


    def save_ck(self):
        """
        Save training checkpoint to the log directory.
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience': self.patience,
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        Train model for one epoch on the training-set.
        """
        start_time = time()
        self.model.train()

        times = []
        train_losses = []
        for step, sample in enumerate(self.train_loader):
            t = time()

            self.optimizer.zero_grad()

            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            y_pred = self.model.forward(x)
            # loss = nn.MSELoss()(y_pred, y_true)
            loss = nn.BCELoss()(y_pred, y_true)
            loss.backward()
            train_losses.append(loss.item())

            # update scheduler and optimizer
            self.scheduler.step(step=step, epoch=self.epoch)
            self.optimizer.step(None)

            # print progress bar
            times.append(time() - t)
            c1 = self.cnf.log_each_step
            c2 = (not c1 and self.progress_bar.progress == 1)
            if c1 or c2:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'\r{self.progress_bar} '
                    f'│ Loss: {np.mean(train_losses):.6f} '
                    f'│ LR: {lr:06f}', end=''
                    )
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses)
        self.sw.add_scalar(
            tag='train_loss', scalar_value=mean_epoch_loss,
            global_step=self.epoch
        )

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def IoU(self, mask1, mask2):
        """
        Compute the Intersection over Union (IoU) between two masks.

        :param mask1: numpy array of shape (B, H, W, C)
        :param mask2: numpy array of shape (B, H, W, C)

        :return: intersection / union: numpy array of shape (B,) with IoU values for each couple of masks in the batch
        """
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        # print(mask1.shape, mask2.shape)
        intersection = np.logical_and(mask1, mask2).sum(axis=(1, 2, 3))
        area1 = mask1.sum(axis=(1, 2, 3))
        area2 = mask2.sum(axis=(1, 2, 3))
        union = area1 + area2 - intersection
        union = np.where(union <= 0, 1e-5, union)

        return intersection / union


    def validate(self):
        """
        Test model on the validation set.
        """
        self.model.eval()

        t = time()
        val_losses = []
        for step, sample in enumerate(self.val_loader):
            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            y_pred = self.model.forward(x)
            y_pred = self.post_proc.binary(y_pred)

            # loss = nn.MSELoss()(y_pred, y_true)
            # val_losses.append(loss.item())

            iou = self.IoU(y_pred, y_true)  # size = (B,)
            val_losses.append(iou)

            # draw results for this step in a 3 rows grid:
            # row #1: input (x)
            # row #2: predicted_output (y_pred)
            # row #3: target (y_true)

            # get the y_pred from c = 1 to c = 3
            y_pred = y_pred.expand(-1, 3, -1, -1).to(self.cnf.device)
            y_true = y_true.expand(-1, 3, -1, -1).to(self.cnf.device)
            grid = torch.cat([x, y_pred, y_true], dim=0)
            grid = tv.utils.make_grid(
                grid, normalize=True, value_range=(0, 1),
                nrow=x.shape[0]
            )
            self.sw.add_image(
                tag=f'results_{step}',
                img_tensor=grid, global_step=self.epoch
            )

        val_losses = torch.cat(val_losses)  # concatenate all the IoU values
        # save best model
        # mean_val_loss = np.mean(val_losses)
        mean_val_loss = val_losses.mean().item()

        first_time = self.best_val_loss is None
        if first_time or (mean_val_loss < self.best_val_loss):
            self.best_val_loss = mean_val_loss
            self.patience = self.cnf.max_patience
            self.model.save_w(self.log_path / 'best.pth', cnf=self.cnf.dict)
        else:
            self.patience = self.patience - 1

        # log val results
        print(
            f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f}'
            f' │ patience: {self.patience}'
            f' │ T: {time() - t:.2f} s'
            )

        # log val loss / val metric
        self.sw.add_scalar(
            'val_loss', scalar_value=mean_val_loss,
            global_step=self.epoch
        )

        # log patience
        self.sw.add_scalar(
            'patience', scalar_value=self.patience,
            global_step=self.epoch
        )

        if self.patience == 0:
            print('\n--------')
            print(f'[■] Done! -> `patience` reached 0.')
            self.save_ck()
            exit(0)


    def run(self):
        """
        Start model training procedure:
            ->> train ->> val ->> checkpoint ->> repeat
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.no_grad():
                self.validate()

            self.epoch += 1
            self.save_ck()
