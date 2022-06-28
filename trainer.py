import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import pytorch_lightning as pl
import nibabel as nib

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
)
from monai.data import (
    DataLoader,
    decollate_batch,
)

class Train(pl.LightningModule):
    """
    The following training script uses the Monai framework for pre processing and post-processing; 
    https://github.com/Project-MONAI/tutorials
    """
    def __init__(self, model, num_classes, lr, weight_decay, sliding_inference, ROI, image_format, labels, output_dir):
        super().__init__()
        self._model = model        
        self.num_classes = num_classes
        self.lr = lr
        self.wd = weight_decay
        self.sliding_inference = sliding_inference
        self.ROI = ROI
        self.image_format = image_format
        self.output_dir = output_dir
        self.imagespath = os.path.join(self.output_dir, 'images')
        self.resultdir = os.path.join(self.output_dir, 'results')
        self.labels = labels
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.dictresults = {'dice':[], 'hd': [], 'asd': []}
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.epoch_loss_values = []


    def forward(self, input):
        quant,  latents, loss, info = self._model(input)
        return quant, loss, latents, info

    def inference(self, input):
        quant, latents, loss, info = self._model(input)
        return quant

    def save_images(self, outputs, imagespath, image_format, batch_idx, image_type):
        for i in range(len(outputs)):
            output = torch.squeeze(outputs[i])
            filename =  os.path.join(imagespath,'{image_t}{img}batch{batch}.{filetype}'.format(image_t=image_type, img = i,batch=batch_idx, filetype = 'nii'))# if self.image_format == 'nifti' else 'png'))
            if image_format == 'nifti':
               output = output.cpu().numpy().astype(np.float32)
               affine = np.eye(4)
               output = nib.Nifti1Image(output, affine)
               nib.save(output, filename)
            else:
               torchvision.utils.save_image(output, filename) 
                
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output, embloss, latents, info = self.forward(images)
        loss = self.loss_function(output, labels) +embloss
        print('train Loss: %.3f' % (loss), 'codebook mean distance: %.6f' % (info[2]), 'codebook mean variance: %.6f' % (info[3]))
        tensorboard_logs = {"train_loss": loss.item(), "mean_cb_distance": info[2].item(), "mean_cb_variance": info[3].item(),}
        self.log("train_loss", loss.item())
        self.log("codebook",  {"mean_cb_distance": info[2].item(), "mean_cb_variance": info[3].item()})
        return {"loss": loss} 

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'].contiguous(), batch['label'].contiguous()
        if self.sliding_inference == True:
           outputs = sliding_window_inference(images, self.ROI, 4, self.inference)
        else:
           outputs = self.inference(images)

        loss = self.loss_function(outputs, labels)
        print('val Loss: %.3f' % (loss))
        self.log("val_loss", loss)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels= [self.post_label(i) for i in decollate_batch(labels)]
        dice = compute_meandice(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        hd = compute_hausdorff_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        asd = compute_average_surface_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        self.log("dice_loss", dice)
        return {"val_loss": loss,"dice": dice, "hd": hd, "asd": asd, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items, dice, hd, asd = 0, 0, 0, 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            dice += output["dice"].sum().item()
            hd += output["hd"].sum().item()
            asd += output["asd"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        mean_dice = torch.tensor(dice / num_items)
        mean_hd = torch.tensor(hd / num_items)
        mean_asd = torch.tensor(asd / num_items)
        self.log("mean_val_loss", mean_val_loss)
        self.log("mean_dice", mean_dice)
        if mean_dice > self.best_val_dice:
            self.best_val_dice = mean_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch}"
            f"current mean dice: {mean_dice:.4f}"
            f"current hd: {mean_hd:.4f}"
            f"current asd: {mean_asd:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return  {"mean_val_loss": mean_val_loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        if self.sliding_inference == True:
           outputs = sliding_window_inference(images, self.ROI, 4, self.inference)
        else:
           outputs = self.inference(images)

        loss = self.loss_function(outputs, labels)
        print('test Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        dice = compute_meandice(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        hd = compute_hausdorff_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        asd = compute_average_surface_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels, dim=0), include_background=False)
        print(
                f"dice: {torch.mean(dice):.4f}"
                f"hd: {torch.mean(hd):.4f}"
                f"asd: {torch.mean(asd):.4f}"
        )
        self.save_images(outputs, self.imagespath,  self.image_format, batch_idx, 'seg')
        self.save_images(labels, self.imagespath,  self.image_format, batch_idx, 'label')
        self.save_images(images, self.imagespath,   self.image_format, batch_idx, 'image')


        return {'dice': dice, 'hd': hd, 'asd': asd}
    def test_epoch_end(self, outputs):
        for output in outputs:
            self.dictresults['dice'].append(output['dice'])
            self.dictresults['asd'].append(output['asd'])
            self.dictresults['hd'].append(output['hd'])
        self.dictresults['dice'] = torch.mean(torch.cat(self.dictresults['dice'], dim = 0),0).squeeze().tolist()
        self.dictresults['asd'] = torch.mean(torch.cat(self.dictresults['asd'], dim = 0),0).squeeze().tolist()
        self.dictresults['hd'] = torch.mean(torch.cat(self.dictresults['hd'], dim = 0),0).squeeze().tolist()
        
        df = pd.DataFrame.from_dict(self.dictresults, orient="index", columns = self.labels)
        df.to_csv(os.path.join(self.resultdir, 'testresults.csv'))
                
        return self.dictresults
