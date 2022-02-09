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
#from convit import VisionTransformer
from timm.models.efficientnet import EfficientNet
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from transformers import ViTForImageClassification
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
import torch.distributed as dist
from mae import MAEVisionTransformers as MAE
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
img_data_dir = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
def cosine_learning_rate(lr, total_epochs,warm_epochs, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate
    """

    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1/2 * (1 + math.cos(batch_iter * math.pi /
                                     ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_adj
    return lr * lr_adj
def build_mask(mask_index, patch_size, img_size):
    num_pathces = img_size // patch_size
    mask_map = torch.zeros((img_size, img_size)).float()
    # reshape the h w -> n c
    mask_map = rearrange(mask_map, '(h p1) (w p2) -> (h w) (p1 p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size)
    mask_index = [index-1 for index in mask_index ]
    mask_map[mask_index] = 1.
    # reshape the n c -> h w
    mask_map = rearrange(mask_map, '(h w) (p1 p2) -> (h p1) (w p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size).cuda()
    return mask_map

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask_map):
        # print(pred.shape)
        # print(target.shape)
        # print(mask_map.shape)
        pred = pred * mask_map
        target = target * mask_map
        loss = F.mse_loss(pred, target)
        return loss
class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, pseudo_rgb, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=True, pseudo_rgb=pseudo_rgb)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

# Press the green button in the gutter to run the script.



class maskViT(pl.LightningModule):
    def __init__(self, num_classes, train_csv, batch_size):
        super().__init__()
        self.automatic_optimization = False
        data = pd.read_csv(train_csv)
        self.trainlen = len(data)
        self.num_classes = num_classes
        self.model = MAE(
            img_size=224,
            patch_size=16,
            encoder_dim=768,
            encoder_depth=12,
            encoder_heads=12,
            decoder_dim=512,
            decoder_depth=8,
            decoder_heads=16,
            mask_ratio=0.75
        )
        ckpt = torch.load('/vol/biomedic3/as217/medicaltransformers/vit-mae_losses_0.20102281799793242.pth', map_location="cpu")[
            'state_dict']
        self.model.load_state_dict(ckpt, strict=True)
        self.bn = int(math.ceil(self.trainlen/batch_size))
        #    self.model = DataParallel(self.model,
         #                    device_ids=-1,
          #                   find_unused_parameters=True)


    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        #optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        optimizer = AdamW(
            params_to_update,
            lr=1.5e-4,
            weight_decay=0.05,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer= optimizer, num_warmup_steps=  self.bn *30, num_training_steps= self.trainlen*300, num_cycles = 0.5)

        return [optimizer], [scheduler]

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        outputs, mask_index = self.model(img)
        mask = build_mask(mask_index, 16, 224)
        criterion = MSELoss()
        losses = criterion(outputs, img, mask)

        return losses

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss = self.process_batch(batch)
        #self.log('train_loss', loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print('Train Loss: %.3f'%(loss))
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()



        # step every `n` batches
        if (batch_idx + 1) % 1 == 0:
            sch.step()
        my_lr = sch.get_lr()







        #grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
       # self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)
        print('val Loss: %.3f'%(loss))

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()

def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = CheXpertDataModule(csv_train_img='/vol/biomedic3/as217/medicaltransformers/full_sample_train.csv',
                              csv_val_img='/vol/biomedic3/as217/medicaltransformers/full_sample_val1.csv',
                              csv_test_img='/vol/biomedic3/as217/medicaltransformers/full_sample_test1.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # mode
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
   # torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False
    model_type = maskViT
    model = model_type(num_classes=num_classes, batch_size=batch_size, train_csv='/vol/biomedic3/as217/medicaltransformers/full_sample_train.csv')
    total_rank = torch.cuda.device_count()
    print('rank: {} / {}'.format(0, total_rank))
    #dist.init_process_group(backend='nccl', rank=0, world_size=2)
    #torch.cuda.set_device(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")
    model.to(device)
    #model = DataParallel(model,
    #                         device_ids=[0,1],
     #                        find_unused_parameters=True)
    #model = model1.load_from_checkpoint(
    ##    '/Users/ainkaransanthirasekaram/Desktop/CheXpert-v1.0preproc_224x224/ViT-all/version_66/checkpoints/epoch=43-step=67099.ckpt',
     #    num_classes=num_classes)

    #model = model1.load_from_checkpoint(
     #   '/Users/ainkaransanthirasekaram/Desktop/Chexvitssl/ViT-all/version_68/checkpoints/epoch=94-step=72484.ckpt',
     #   num_classes=num_classes)


    # Create output directory
    out_name = 'ViT-all'
    out_dir = 'outputfinet2/disease/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')


    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        #num_nodes=1,
        #strategy="ddp",
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('output2/disease1', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)


    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    #model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_val1.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_test1.csv'), index=False)


if    __name__ == '__main__':
      parser = ArgumentParser()
      parser.add_argument('--gpus', default=1)
      parser.add_argument('--dev', default=0)
      args = parser.parse_args()

      main(args)
