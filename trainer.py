import matplotlib.pyplot as plt
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
from post_processing import binarize
from progress_bar import ProgressBar
from scheduler import LRScheduler

def decode_mask(img):
    #type: (torch.Tensor) -> torch.Tensor
    '''
    Decodes the image with n channels to a single channel grayscale mask
    :param img: tensor with n channels
    :return: img_decoded: single channel mask torch tensor in uint8
    '''
    background_mask = torch.all(img == 0, dim=1) #shape = (B,1,H,W)
    back = torch.zeros_like(img, dtype=torch.int8).sum(dim=1) #shape = (B,1,H,W)
    back[background_mask] = -1 #background pixels mapped to -1
    img = img.argmax(dim=1) + back #pixels ==0 are now class 0 (not background) and pixels == -1 are background
    img = (img + 1) * 255 / 10 #background is set from -1 to 0 and class i is set to
    #add channel dim (B, 1, H, W) with values in range [0, 255]
    return img.unsqueeze(1).to(torch.uint8)

def get_color_map(num_classes = 10):
    #type: (int) -> torch.Tensor
    """
    Returns a colormap with distinct colors for each class
    :param num_classes: total number of classes
    :return: colormap with distinct colors for each class
    """
    colormap = plt.get_cmap('tab10', num_classes)
    cmap = (colormap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    cmap = torch.from_numpy(cmap).permute(2, 0, 1)

def decode_mask_rgb(img, num_classes = 10, background_color = (0, 0 , 0), colormap=None):
    """
    Decode an image with n channels into an RGB mask with distinct colors for each class.
    :param img: immagine con n canali (B, C, H, W)
    :param num_classes: numero totale di classi
    :param background_color: colore RGB per i pixel di background
    :param colormap: colormap from matplotlib with distinct colors for each class (use get_color_map helper function)
    :return: immagine RGB con colori distinti per classe
    """
    background_mask = torch,all(img == 0, dim = 1) #identify background pixels
    class_map = img.argmax(dim=1) #determine the class with the highest probability for each pixel
    class_map[background_mask] = -1 #set -1 for background pixels

    #create a tensor rgb_colors used as a colormap during decoding
    if colormap is None:
        colormap = get_color_map(num_classes).to(img.device)
    class_colors = torch.tensor(colormap, dtype = torch.uint8, device = img.device)
    background_color = torch.tensor(background_color, dtype = torch.uint8, device = img.device)
    rgb_colors = torch.cat([background_color, class_colors], dim=0)
    # set background color to background pixels with fancy indexing
    decoded_img = rgb_colors[class_map + 1]  #color_map+1 because currently we have class_map [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] we want to map it to range [0, 10]
    return decoded_img

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
    inters = torch.logical_and(masks_pred, masks_true).sum((2, 3))
    union = torch.logical_or(masks_pred, masks_true).sum((1, 2, 3))
    return torch.where(union != 0, inters / union, 0)

def get_batch_iou_for_classes(masks_pred, masks_true):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Compute the intersection over union between two batches of binary masks.

    :param masks_pred: predicted binary masks
        ->> shape: (B, C, H, W); values: bin{0, 1}
    :param masks_true: target binary masks
        ->> shape: (B, C, H, W); values: bin{0, 1}
    :return: IoU value for each element in the batch
        ->> shape: (B,C); values: range[0, 1]
    """
    inters = torch.logical_and(masks_pred, masks_true).sum((2, 3))
    union = torch.logical_or(masks_pred, masks_true).sum((2, 3))
    return torch.where(union != 0, inters / union, 0)

class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        """
        :param cnf: configuration object that contains all the
            hyper-parameters and paths.
        """

        self.cnf = cnf
        self.cmap = get_color_map()
        # init model
        # self.model = DummyModel()
        self.model = UNet(input_channels=3, output_channels=10)
        self.model = self.model.to(cnf.device)

        # init optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(), lr=cnf.lr
        )

        # init train loader
        training_set = CocoDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True,
            worker_init_fn=training_set.wif,
        )

        # init val loader
        val_set = CocoDS(cnf, mode='val')
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=cnf.batch_size,
            num_workers=1, shuffle=False, pin_memory=True,
            worker_init_fn=val_set.wif_val,
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
            iou = get_batch_iou_for_classes(y_pred, y_true) # size = (B,C)
            acc = torch.where(iou > 0.5, 1, 0)
            #accumulates the iou and accuracy for each batch in the validation set
            val_iou.append(iou)
            val_acc.append(acc)

            # draw results for this step in a 3 rows grid:
            # row #1: input (x)
            # row #2: predicted_output (y_pred)
            # row #3: target (y_true)
            #reshaping tensors for visualization purposes
            # decoding mask and target mask from c = 10 to c= 1
            y_true = decode_mask(y_true)
            y_pred = decode_mask(y_pred)
            # y_true = y_true.sum(dim=1, keepdim=True)
            # from c = 1 to c = 3 for visualization purposes
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

        val_iou_classes = torch.cat(val_iou).mean(dim=0)  # concatenate all the IoU values and take the mean of the whole validation set for each class
        val_acc_classes = torch.cat(val_acc).to(torch.float32).mean(dim=0) #non sicuro che serva .to() val acc è gia un tensore!(rivedere perchè è qua)
        # save best model
        mean_iou = val_iou_classes.mean().item() #mean of the mean IoU for each class
        mean_acc = val_acc_classes.mean().item() #mean of the mean accuracy for each class

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
