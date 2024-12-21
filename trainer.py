import numpy as np
import torch
import torchvision as tv
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import time
from conf import Conf
# from dataset.dummy_ds import DummyDS
from dataset.coco_ds import CocoDS
from models import UNet
from post_processing import binarize, get_color_map, decode_mask_rgb
from progress_bar import ProgressBar
from scheduler import LRScheduler
from visual_utils import decode_and_apply_mask_overlay

# def get_batch_iou(masks_pred, masks_true):
#     # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
#     """
#     Compute the intersection over union between two batches of binary masks.
#
#     :param masks_pred: predicted binary masks
#         ->> shape: (B, 1, H, W); values: bin{0, 1}
#     :param masks_true: target binary masks
#         ->> shape: (B, 1, H, W); values: bin{0, 1}
#     :return: IoU value for each element in the batch
#         ->> shape: (B,); values: range[0, 1]
#     """
#     inters = torch.logical_and(masks_pred, masks_true).sum((2, 3))
#     union = torch.logical_or(masks_pred, masks_true).sum((1, 2, 3))
#     return torch.where(union != 0, inters / union, 0)

def get_batch_iou_multiclass(masks_pred, masks_true):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Compute the intersection over union between two batches of binary masks.

    :param masks_pred: predicted binary masks
        ->> shape: (B, C, H, W); values: bin{0, 1}
    :param masks_true: target binary masks
        ->> shape: (B, C, H, W); values: bin{0, 1}
    :return: IoU value for each element in the batch, where IoU = mask_pred ∩ mask_true / mask_pred ∪ mask_true
        ->> shape: (B,C); values: range[0, 1]
    """
    assert masks_pred.shape == masks_true.shape, "Predicted and target masks must have the same shape"

    inters = torch.logical_and(masks_pred, masks_true).sum((2, 3))
    union = torch.logical_or(masks_pred, masks_true).sum((2, 3))
    #tensore B,C raprresentante gli indici dei canali della masks_true tutti nulli
    zero_channel = torch.all(masks_true == 0, dim=(2, 3))
    IoUs = torch.where(union != 0, inters / union, 0)
    #setto a nan gli IoU relativi ai canali della masks_true tutti nulli
    IoUs[zero_channel] = torch.tensor(float('nan'))
    return IoUs


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        """
        :param cnf: configuration object that contains all the hyperparameters and paths.
        """

        self.cnf = cnf
        # init model
        self.model = UNet(input_channels=3, num_classes=cnf.num_classes)
        self.model = self.model.to(cnf.device)
        self.cmap = get_color_map(cnf.num_classes).to(cnf.device) #color map for the visualization of the results passed as args to decode_mask_rgb in validation
        # init optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(), lr=cnf.lr
        )


        #if collate_fn=training_set.collate_fn, data augmentation is applied at batch level on device
        #take in account that currently DA is made in coco_ds.py, during data loading
        training_set = CocoDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True,
            worker_init_fn=training_set.wif, collate_fn=training_set.collate_fn
        )

        # init val loader
        val_set = CocoDS(cnf, mode='val', data_augmentation=False)
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=cnf.batch_size,
            num_workers=1, shuffle=False, pin_memory=True,
            worker_init_fn=val_set.wif_val
        )

        # self.post_proc = PostProcessor(out_ch_order='RGB')

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
        self.best_val_iou = None
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
            self.best_val_iou = ck['best_val_iou']
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
            'best_val_iou': self.best_val_iou,
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
            # loss = nn.CrossEntropyLoss()(y_pred, y_true)
            loss = nn.BCELoss()(y_pred, y_true) #y_true è binaria, y_pred è una mappa di probabilità per ogni pixels
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


    def validate(self):
        """
        Test model on the validation set.
        """
        self.model.eval()

        t = time()
        val_iou = []
        val_acc = []

        for step, sample in enumerate(self.val_loader):
            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            y_pred = self.model.forward(x)
            y_pred = binarize(y_pred)
            iou = get_batch_iou_multiclass(y_pred, y_true) # size = (B,C)
            acc = torch.where(iou > 0.5, 1, 0)
            #accumulates the iou and accuracy for each batch in the validation set
            val_iou.append(iou)
            val_acc.append(acc)

            #reshaping tensors for visualization purposes
            # from c = 10 to c = 3 for visualization purposes
            # y_pred = decode_mask_rgb(y_pred, colormap=self.cmap).float() / 255.0
            # y_true = decode_mask_rgb(y_true, colormap=self.cmap).float() / 255.0
            y_pred = decode_and_apply_mask_overlay(x, y_pred)
            y_true = decode_and_apply_mask_overlay(x, y_true)
            # draw results for this step in a 3 rows grid:
            # row #1: input (x)
            # row #2: predicted_output (y_pred)
            # row #3: target (y_true)
            grid = torch.cat([x, y_pred, y_true], dim=0)
            grid = tv.utils.make_grid(
                grid, normalize=True, value_range=(0, 1),
                nrow=x.shape[0]
            )
            self.sw.add_image(
                tag=f'results_{step}',
                img_tensor=grid, global_step=self.epoch
            )

        val_iou_classes = torch.cat(val_iou).nanmean(dim=0)  # concatenate all the IoU values and take the mean of the whole validation set for each class
        val_acc_classes = torch.cat(val_acc).to(torch.float32).mean(dim=0) #non sicuro che serva .to() val acc è gia un tensore!(rivedere perchè è qua)
        # save best model
        mean_iou = val_iou_classes.nanmean().item() #mean of the mean IoU for each class
        mean_acc = val_acc_classes.nanmean().item() #mean of the mean accuracy for each class

        first_time = self.best_val_iou is None
        if first_time or (mean_iou > self.best_val_iou):
            self.best_val_iou = mean_iou
            self.patience = self.cnf.max_patience
            self.model.save_w(self.log_path / 'best.pth', cnf=self.cnf.dict)
        else:
            self.patience = self.patience - 1

        # log val results
        print(
            f'\t● IoU for each class: {[f"{iou:.3f}" for iou in val_iou_classes.tolist()]}\n'
            f'\t● AVG IoU on VAL-set: {mean_iou:.6f}'
            f' │ AVG Accuracy on VAL-set: {mean_acc:.6f}'
            f' │ patience: {self.patience}'
            f' │ T: {time() - t:.2f} s'
        )

        # log val loss / val metric
        self.sw.add_scalar(
            'iou', scalar_value=mean_iou,
            global_step=self.epoch
        )

        self.sw.add_scalar(
            'accuracy', scalar_value=mean_acc,
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

# if __name__ == '__main__':
#
#     #Iou multiclasses test
#     masks_pred = torch.tensor([
#         [[[0, 1], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]]
#     ], dtype=torch.uint8)  # Shape: (1, 3, 2, 2)
#
#     masks_true = torch.tensor([
#         [[[0, 1], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]]
#     ], dtype=torch.uint8)  # Shape: (1, 3, 2, 2)
#
#     # Calcolo dell'IoU
#     iou = get_batch_iou_multiclass(masks_pred, masks_true)
#     print("IoU:", iou)
#     print("mean IoU:", iou.nanmean().item())