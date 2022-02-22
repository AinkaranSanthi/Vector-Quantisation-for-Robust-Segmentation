import os
# from numpy.core.numeric import full
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from pandas.core.dtypes import base
import torch
import tqdm
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
# import pickle
from torch.utils.data.dataset import Dataset
import pytorch_lightning
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import math
import torchvision
from skimage import measure
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric,compute_meandice,  compute_hausdorff_distance, compute_average_surface_distance
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
from convnets import GumbelUNet2Dpos
#import torchio as tio
from PIL import Image
from transformers.optimization import get_cosine_schedule_with_warmup
# from loaders.ChestXray_NIHCC import pre_processing

def hist_specification(hist, bit_output, min_roi, max_roi, flag_jsrt, gamma):

    cdf = hist.cumsum()
    cdf = np.ma.masked_equal(cdf, 0)

    # hist sum of low & high
    hist_low = np.sum(hist[:min_roi+1]) + flag_jsrt
    hist_high = cdf.max() - np.sum(hist[max_roi:])

    # cdf mask
    cdf_m = np.ma.masked_outside(cdf, hist_low, hist_high)

    # build cdf_modified
    if not (flag_jsrt):
        cdf_m = (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    else:
        cdf_m = (bit_output-1) - (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m.astype('float32'), 0)

    # gamma correction
    cdf = pow(cdf/(bit_output-1), gamma) * (bit_output-1)

    return cdf


def pre_processing(images, flag_jsrt=0, rescale_bit=8, gamma=.5):


    # histogram
    num_out_bit = 1<<rescale_bit
    num_bin = images.max()+1

    # histogram specification, gamma correction
    hist, bins = np.histogram(images.flatten(), num_bin, [0, num_bin])
    cdf = hist_specification(hist, num_out_bit, images.min(), num_bin, flag_jsrt, gamma)
    images = cdf[images].astype('float32')

    # normalise intensities between -1 and 1
    a = 2/255.
    b = -1.
    images = images * a + b
    return images
class JSRTDataset(Dataset):
    def __init__(self, base_path="/vol/biodata/data/chest_xray/JSRT",
                 csv_path="/vol/biodata/data/chest_xray/JSRT",
                 csv_name="jsrt_metadata_with_masks.csv", # TODO: separate into groups
                 target_size=256,
                 input_channels=1, labels=('right lung', 'left lung' ),
                 rescale_bit=8,
                 gamma=0.5,
                 **kwargs):
        self.df = pd.read_csv(os.path.join(csv_path, csv_name))
        self.base_path = base_path
        self.labels = labels if labels is not None else []
        self.input_channels = input_channels
        self.target_size = target_size
        self.rescale_bit = rescale_bit
        self.gamma = gamma

    def __getitem__(self, index):
        i = self.df.index[index]
        path = os.path.join(self.base_path, self.df.loc[i, "path"])
        img = Image.open(path)
        img = img.resize((self.target_size, self.target_size))
        # note that pre_processing knows how to handle jsrt images TODO: work out what they do
        img = pre_processing(np.array(img), flag_jsrt=1, rescale_bit=self.rescale_bit, gamma=self.gamma)
        img = np.stack([img] * self.input_channels, 0)
        if len(self.labels) > 0:
            labels = []
            for l in self.labels:
                path = os.path.join(self.base_path, "SCR/masks", l, self.df.loc[i, 'id']+ ".gif")
                label = Image.open(path)
                label = label.resize((self.target_size, self.target_size))
                # the interpolation uses a nearest filter, yet it produces some smoothing
                # TODO: check why
                label = ((np.array(label) / 255) > .5).astype(int)
                labels.append(label)
            labels = np.stack(labels, axis=0)
            label = (labels.sum(0) == 0).astype(int)
            labels = np.concatenate([label[np.newaxis], labels], 0)
            return torch.Tensor(img).float(), torch.Tensor(labels).float()
        return torch.Tensor(img).float()

    def __len__(self):
        return len(self.df)

class JSRTSemanticGANDataset(JSRTDataset):
    def __init__(self,  base_path="/vol/biodata/data/chest_xray/JSRT",
                 csv_path="/vol/biodata/data/chest_xray/JSRT",
                 csv_name="jsrt_metadata_with_masks.csv", # TODO: separate into groups
                 target_size=256,
                 input_channels=3, labels=('left lung', 'right lung'),
                 rescale_bit=8,
                 gamma=0.5):
        super().__init__(base_path, csv_path, csv_name, target_size, input_channels, labels, rescale_bit, gamma)

    def __getitem__(self, i):
        items = super().__getitem__(i)
        if len(self.labels)>0:
            return {"image": items[0], "mask":items[1]}
        else:
            return {'image': items}

class JSRTDataloader(LightningDataModule):
    def __init__(self, batch_size, csv_names,base_paths, csv_paths, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets1 = JSRTDataset(csv_name=csv_names[0],base_path = base_paths[0], csv_path = csv_paths[0],  **kwargs) 
        self.datasets2 = NIHDataset(csv_name=csv_names[1],base_path = base_paths[1], csv_path = csv_paths[1],  **kwargs)
        self.datasets3 = NIHDataset(csv_name=csv_names[2],base_path = base_paths[1], csv_path = csv_paths[1],  **kwargs)
    def train_dataloader(self):
        return DataLoader(self.datasets1, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets2, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets3, batch_size=self.batch_size, num_workers=self.num_workers)


class NIHDataset(Dataset):
    def __init__(self, base_path, csv_path, csv_name, input_channels= 1,
            k_shots=None,
                 gamma=0.5,
                 target_size=256,
                 with_mask=True,
                 two_masks=True,
                 diag=False,
                 **kwargs):
        super().__init__()
        self.df = pd.read_csv(os.path.join(csv_path, csv_name))
        self.base_path = base_path
        self.input_channels = input_channels

        self.gamma = gamma
        self.target_size = target_size
        self.with_mask = with_mask
        self.two_masks = two_masks
        self.diag = diag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        i = self.df.index[i]
        path = os.path.join(self.base_path, self.df.loc[i, 'scan'])
        img = Image.open(path)
        img = img.resize((self.target_size, self.target_size))
        img = np.array(img)
        if len(img.shape)!=2:
            img = img[:, :, 0]
        img = pre_processing(img, flag_jsrt=0, rescale_bit=8, gamma=self.gamma)
        # replicate the image to have 'input_channels' num of channels
        img = np.stack([img] * self.input_channels, 0)
        if self.with_mask:
            if self.two_masks:
                return torch.Tensor(img).float(), torch.Tensor(self.gettwomasks(i)).float()
            else:
                return torch.Tensor(img).float(), torch.Tensor(self.getmask(i)).float()
        elif self.diag:
            return torch.Tensor(img).float(), torch.Tensor(self.getdiaglabel(i)).float()
        else:
            return torch.Tensor(img).float()


    def getmask(self, i):
        path = os.path.join(self.base_path, self.df.loc[i, 'mask'])
        mask = Image.open(path)
        mask = mask.resize((self.target_size, self.target_size))
        mask = np.array(mask)
        mask = ((np.array(mask) / 255) > .5).astype(int)
        mask = np.stack([1-mask, mask])
        return mask



    def gettwomasks(self, i):
            path = os.path.join(self.base_path, self.df.loc[i, 'mask'])
            mask = Image.open(path)
            mask = mask.resize((self.target_size, self.target_size))
            mask = np.array(mask)
            mask = ((np.array(mask) / 255) > .5).astype(int)
            # separate masks
            # we swap axes so that the left lung is the first connected region to be encountered on the y axis
            mask = np.swapaxes(mask, 0, 1)
            mask = measure.label(mask)
            mask = np.swapaxes(mask, 1, 0)
            mask = np.stack([mask == i for i in range(3)])
            return mask

    def getdiaglabel(self, i):
            label = self.df.loc[i, 'Finding Labels'] == "No Finding"
            label = [label, not label]
            return label

class NIHDataLoader(LightningDataModule):
        """
        A data module that gives the training, validation and test data for a simple 1-dim regression task.
        """

        def __init__(self,
                     batch_size, csv_names, num_workers=8, **kwargs
                     ):
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.datasets = [NIHDataset(csv_name=csv_names[i], **kwargs) for i in range(3)]

        def train_dataloader(self, *args, **kwargs):
            return DataLoader(self.datasets[0], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        def val_dataloader(self, *args, **kwargs):
            return DataLoader(self.datasets[1], batch_size=self.batch_size, num_workers=self.num_workers)

        def test_dataloader(self, *args, **kwargs):
            return DataLoader(self.datasets[2], batch_size=self.batch_size, num_workers=self.num_workers)

class Net(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.automatic_optimization = False
       ### data = pd.read_csv(train_csv)
        #self.trainlen = len(data)
        self.num_classes = num_classes
        self._model = GumbelUNet2Dpos(
            inputchannels=1,
            num_classes = 3,
            channels=32,
            dropout=0.0,
            n_embed=1024,
            embed_dim=256
        )
        self.loss_function = DiceCELoss(to_onehot_y=False, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.post_label = AsDiscrete(to_onehot=3)
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.dictresults = {'dice':[], 'hd': [], 'asd': []}
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 50
        self.check_val = 5
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []
        
        #self.getalpha = lambda x: np.cos(x/self.max_epochs * np.pi/2.0)
        #selxf.bn = int(math.ceil(self.trainlen/batch_size))

    def forward(self, input):
        if self.current_epoch<(0.1 * self.max_epochs):
            alpha = 1
        elif self.current_epoch>(0.4 * self.max_epochs):
            alpha = 0
        else:
             alpha = np.cos((self.current_epoch-(self.max_epochs*0.1))/(self.max_epochs * np.pi/2.0))
        quant, loss, latents = self._model(input, 0.5)
        return quant, loss

    def forward1(self, input):
        if self.current_epoch<(0.1 * self.max_epochs):
            alpha = 1
        elif self.current_epoch>(0.4 * self.max_epochs):
            alpha = 0
        else:
             alpha = np.cos((self.current_epoch-(self.max_epochs*0.1))/(self.max_epochs * np.pi/2.0))
        quant, loss, latents = self._model(input, 0.5)
        return quant

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def configure_optimizers(self):
        itern = self.max_epochs * 15
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer= optimizer, num_warmup_steps= itern *0.4 , num_training_steps= itern, num_cycles = 0.5)
        return [optimizer], [scheduler]
       # return optimizer
    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        output, embloss = self.forward(images)
        loss = self.loss_function(output, labels) +embloss
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()


        print('train Loss: %.3f' % (loss))
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
        print('val Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        #labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels  )
        hd = compute_hausdorff_distance(y_pred=torch.stack(outputs,dim=0), y=labels)
        asd = compute_average_surface_distance(y_pred=torch.stack(outputs,dim=0), y=labels)
        return {"val_loss": loss,"hd": hd,"asd": asd, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items, hd, asd = 0, 0, 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            hd += output["hd"].sum().item()
            asd += output["asd"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        mean_hd = torch.tensor(hd / num_items)
        mean_asd = torch.tensor(asd / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
            "val_hd": mean_hd,
            "val_asd": mean_asd,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"current hd: {mean_hd:.4f}"
            f"current asd: {mean_asd:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
        print('test Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
       # labels = [self.post_label(i) for i in decollate_batch(labels)]
       # dice1, num_items1, hd1, asd1 = 0, 0, 0, 0
        num_items = len(outputs)
        dice = torch.mean(compute_meandice(y_pred=torch.stack(outputs,dim=0), y=labels))
        hd = torch.mean(compute_hausdorff_distance(y_pred=torch.stack(outputs,dim=0), y=labels  ))
        asd = torch.mean(compute_average_surface_distance(y_pred=torch.stack(outputs,dim=0), y=labels))
        print(
            f"dice: {dice:.4f}"
            f"hd: {hd:.4f}"
            f"asd: {asd:.4f}"
        )

        imagespath = '/vol/biomedic3/as217/vqseg/outputchestseg/gumbelvqJRSTtoNIH/testimagesoutput/'

        dictresults = {'dice':[], 'hd': [], 'asd': []}
        self.dictresults['dice'].append(dice)
        self.dictresults['asd'].append(asd)
        self.dictresults['hd'].append(hd)
        df = pd.DataFrame.from_dict(self.dictresults, orient="index")
        os.makedirs('/vol/biomedic3/as217/vqseg/outputchestseg/gumbelvqJRSTtoNIH/testresults/', exist_ok=True)
        df.to_csv( '/vol/biomedic3/as217/vqseg/outputchestseg/gumbelvqJRSTtoNIH/testresults/result.csv')
        for i in range(len(outputs)):
           torchvision.utils.save_image(outputs[i], os.path.join(imagespath, 'image{img}batch{batch}.png'.format(img = i, batch = batch_idx)))

        return dictresults


# train
if    __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    csv1 = ["JSRT_{}.csv".format(_set) for _set in ['train']]
    csv2 = ["{}_NIH.csv".format(_set) for _set in [ 'val', 'test']]
    net = Net(num_classes=3)
    data = JSRTDataloader(
        batch_size=10,
        num_workers=4,
        base_paths=["/vol/biodata/data/chest_xray/JSRT","/vol/biodata/data/chest_xray/NIH/"],
        csv_paths=["/vol/biomedic3/mmr12/data/chestXR/JSRT/","/vol/biomedic3/mmr12/data/chestXR/NIHCC-CXR14/"],
        csv_names=csv1 + csv2 ,
        input_channels=1
    )

    out_name = 'gumbelvqchestJSRTtoNIH'
    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(monitor="val_dice", mode='min')

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=checkpoint_callback,
        logger=TensorBoardLogger('outputchestseg/gumbelvqJSRTtoNIH', name=out_name),
    )

    trainer.fit(net, data)
    os.makedirs('/vol/biomedic3/as217/vqseg/outputchestseg/gumbelvqJRSTtoNIH/testimagesoutput/ ', exist_ok=True)
    trainer.test(net, data.test_dataloader())
