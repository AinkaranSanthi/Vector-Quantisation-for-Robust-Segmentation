import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from convnets import VQUNet3D, UNet3D
from sklearn.model_selection import KFold
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
import pytorch_lightning as pl
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
from torch.utils.data.dataset import Dataset, Subset
import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule, seed_everything, Trainer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

class Data():
    def __init__(self, fold, numfolds):
        super().__init__()
        self.fold = fold
        self.num_folds = numfolds

    #def prepare_data(self):
        # prepare data
        # data_dir ='/dataset/dataset0/'
        #data_dir = "/Users/ainkaransanthirasekaram/Desktop/RawData/"
        data_dir = "/vol/biomedic3/as217/RawD/RawData/"
        split_JSON = "dataset_1.json"
        datasets = data_dir + split_JSON
        datalist = load_decathlon_datalist(datasets, True, "training")
        self.splits = [split for split in KFold(self.num_folds).split(range(len(datalist)))]
        train_indices, val_indices = self.splits[self.fold]
        self.train_fold = Subset(datalist, train_indices)
        print(len(self.train_fold))
        self.val_fold = Subset(datalist, val_indices)

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                #CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_ds = CacheDataset(
            data=self.train_fold,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=self.val_fold,
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=list_data_collate,
        )
        return val_loader


class Net(pytorch_lightning.LightningModule):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold

        self._model = UNet3D(
            inputchannels=1,
            num_classes = 14,
            channels=16,
            dropout=0.0,
            n_embed=1024,
            embed_dim=256
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.post_label = AsDiscrete(to_onehot=14)
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 300
        self.check_val = 5
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, input):
        quant,  latents = self._model(input)
        return quant

    def forward1(self, input):
        quant, latents = self._model(input)
        return quant




    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 4:
        #     x = x[..., None]
        # x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format).repeat(1, 3, 1, 1, 1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer





    def training_step(self, train_loader, batch_idx):
        images, labels = train_loader["image"], train_loader["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        print('train Loss//fold-%.1f: %.3f' % (self.fold, loss))
        tensorboard_logs = {"train_loss//fold-%.1f"% (self.fold): loss.item()}
        return {"loss": loss, "log": tensorboard_logs}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, val_loader, batch_idx):
        images, labels = self.get_input(val_loader, "image"),  self.get_input(val_loader, "label")
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward1
        )

        loss = self.loss_function(outputs, labels)
        print('val Loss//fold-%.1f: %.3f' % (self.fold, loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.log("val_loss//fold-%.1f" % (self.fold), loss)
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss//fold-%.1f" % (self.fold): loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss//fold-%.1f" % (self.fold)].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice//fold-%.1f" % (self.fold): mean_val_dice,
            "val_loss//fold-%.1f" % (self.fold): mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}


# train
if    __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('--fold', default='1')
        parser.add_argument('--folds', default='5')
        args = parser.parse_args()


        out_name = "useg:fold-%.1f" % (int(args.fold))
        out_dir = 'vqseglogs/unetcv/' + out_name
    # initialise the LightningModule
        net = Net(fold=int(args.fold))
        checkpoint_callback = ModelCheckpoint(monitor="val_loss//fold-%.1f" % (int(args.fold)), mode='min')

    # set up checkpoints
        #checkpoint_callback = ModelCheckpoint(dirpath=root_dir, filename="best_metric_model")

    # initialise Lightning's trainer.
        trainer = pytorch_lightning.Trainer(
            gpus=[0],
            log_every_n_steps=5,
            max_epochs=net.max_epochs,
            check_val_every_n_epoch=net.check_val,
            callbacks=checkpoint_callback,
            logger=TensorBoardLogger('vqseglogs/unetcv/f-%.1f' % (int(args.fold)), name=out_name),
        )
        train, val = Data(fold=int(args.fold),numfolds=int(args.folds)).train_dataloader(), Data(fold=int(args.fold),numfolds=int(args.folds)).val_dataloader()
        trainer.fit(net, train, val)