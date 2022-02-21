import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
from transformers import ViTForImageClassification
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
import nibabel as nib
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandBiasFieldd,

)

from monai.config import print_config
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance,  compute_meandice
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers.optimization import AdamW
from transformers import ViTForImageClassification, BeitFeatureExtractor, BeitForImageClassification, BeitForMaskedImageModeling, BeitModel
import os
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchvision
from convnets import  UNet3Dv2,  VQUNet3Dposv3, GumbelUNet3Dpos
import torchvision.transforms as T
from transformers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from pytorch_lightning.callbacks import LearningRateMonitor
from functools import partial
import argparse
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.helpers import load_pretrained
from torchvision import models
import pytorch_lightning as pl
import pytorch_lightning
from timm.models.efficientnet import EfficientNet
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
import math
from argparse import ArgumentParser
image_size = (224, 224)
num_classes = 14
batch_size = 100
epochs = 300
num_workers = 4



class ProstateDataset(Dataset):
    def __init__(self, csv_file_img, datatype):
        self.data = pd.read_csv(csv_file_img)
        self.datatype = datatype
        self.train_transforms = Compose(
            [   
                LoadImaged(
                    keys = ["image", "label"]
                ),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.5, 0.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
               # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),

                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 64),
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=0,
                    prob=0.10,
                ),
                RandRotated(
                    keys=["image", "label"],
                    range_x=0.2,
                    prob=0.10,
                ),

                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.4),
                RandGaussianNoised(keys = "image", std = 0.05, prob = 0.15),
                RandAdjustContrastd(keys="image",prob = 0.2),
                RandBiasFieldd(keys="image",  prob=0.2),

            ]
        )
        self.val_transforms = Compose(
            [   
                LoadImaged(
                    keys = ["image", "label"]
                ),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.5, 0.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),
                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 64),
                ),
                #ToTensord(keys=["image", "label"]),
            ]
        )

        
        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = self.data.loc[idx, 'images']
            img_label = self.data.loc[idx, 'labels']

            sample = {'image': img_path, 'label': img_label}

            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image'])
        label = torch.from_numpy(sample['label'])
        label = torch.where(label > 0, 1, 0)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        if self.datatype == 'train':
            sample = self.train_transforms(sample)
        else:
            sample = self.val_transforms(sample)

        return sample


class ProstateDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size =batch_size
        self.num_workers = num_workers


        self.train_set = ProstateDataset(self.csv_train_img, datatype='train')
        self.val_set = ProstateDataset(self.csv_val_img, datatype= 'val')
        self.test_set = ProstateDataset(self.csv_test_img, datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
#        self.automatic_optimization = False
       ### data = pd.read_csv(train_csv)
        #self.trainlen = len(data)
        self._model = UNet3Dv2(
        #self._model = GumbelUNet2Dpos(
            inputchannels=1,
            num_classes = 2,
            channels=16,
            dropout=0.0,
            n_embed=1024,
            embed_dim=256
        )
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
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

        #self.bn = int(math.ceil(self.trainlen/batch_size))

    def forward(self, input):
        quant,  latents = self._model(input)
        return quant

    def forward1(self, input):
        quant, latents = self._model(input)
        return quant

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output = self.forward(images)
        loss = self.loss_function(output, labels) 
        print('train Loss: %.3f' % (loss))
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
        print('val Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels1)
        hd = compute_hausdorff_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False)
        asd = compute_average_surface_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False)
        return {"val_loss": loss, "hd": hd, "asd": asd, "val_number": len(outputs)}

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
        images, labels = batch['image'], batch['label']
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
        print('test Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]

        dice = torch.mean(compute_meandice(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False))
        hd = torch.mean(compute_hausdorff_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False))
        asd = torch.mean(compute_average_surface_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False))
        print(
                f"dice: {dice:.4f}"
                f"hd: {hd:.4f}"
                f"asd: {asd:.4f}"
        )

        imagespath = '/vol/biomedic3/as217/vqseg/outputprostatefinal2/unetBMCtoRUM/testimagesoutput/'


        self.dictresults['dice'].append(dice)
        self.dictresults['asd'].append(asd)
        self.dictresults['hd'].append(hd)
        df = pd.DataFrame.from_dict(self.dictresults, orient="index")
        os.makedirs('/vol/biomedic3/as217/vqseg/outputprostatefinal2/unetBMCtoRUM/testresults/', exist_ok=True)
        df.to_csv('/vol/biomedic3/as217/vqseg/outputprostatefinal2/unetBMCtoRUM/testresults/result.csv')
        for i in range(len(outputs)):
                output = torch.squeeze(outputs[i])
                output = torch.argmax(output, dim =0)
                output = output.cpu().numpy().astype(np.float32)
                affine = np.eye(4)
                output = nib.Nifti1Image(output, affine)
                nib.save(output,  os.path.join(imagespath,'seg{img}batch{batch}.nii'.format(img=i,batch=batch_idx)))
                label = torch.squeeze(labels1[i])
                label = torch.argmax(label, dim =0)
                label = label.cpu().numpy().astype(np.float32)
                affine = np.eye(4)
                label = nib.Nifti1Image(label, affine)
                nib.save(label,  os.path.join(imagespath,'label{img}batch{batch}.nii'.format(img=i,batch=batch_idx)) )
                image = torch.squeeze(images[i,:,:,:,:])

                image = image.cpu().numpy().astype(np.float32)
                affine = np.eye(4)
                image = nib.Nifti1Image(image, affine)
                nib.save(image,  os.path.join(imagespath,'image{img}batch{batch}.nii'.format(img=i,batch=batch_idx)))
        return self.dictresults
    # train

if __name__ == '__main__':
        pl.seed_everything(42, workers=True)
        net = Net()
        data = ProstateDataModule(
            batch_size=1,
            num_workers=4,
            csv_train_img="/vol/biomedic3/as217/vqseg/ProstateDomain/Prostatedomaintr1.csv",
            csv_val_img="/vol/biomedic3/as217/vqseg/ProstateDomain/ProstatedomainvalBMC1.csv",
            csv_test_img="/vol/biomedic3/as217/vqseg/ProstateDomain/Prostatedomaints1.csv",
        )

        root_dir = 'outputprostatefinal2/unetBMCtoRUM'
        # set up checkpoints
        #checkpoint_callback = ModelCheckpoint(monitor="val_dice", mode='max')
        checkpoint_callback = ModelCheckpoint(dirpath=root_dir, filename="best_metric_model")
        # initialise Lightning's trainer.
        trainer = pytorch_lightning.Trainer(
              gpus=[0],
              max_epochs=net.max_epochs,
              check_val_every_n_epoch= net.check_val,
              callbacks=checkpoint_callback,
              default_root_dir=root_dir,
        )
        trainer.fit(net, data)
        os.makedirs('/vol/biomedic3/as217/vqseg/outputprostatefinal2/unetBMCtoRUM/testimagesoutput/ ', exist_ok=True)
        net = Net()
        model = net.load_from_checkpoint(os.path.join(root_dir, "best_metric_model-v1.ckpt"))
        model.eval()
        trainer.test(model, data.test_dataloader())
