import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
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
    NormalizeIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandRotate90d,
    RandBiasFieldd,
    ToTensord,
)
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image

"""
The following dataloaders uses the Monai framework for post-processing;
https://github.com/Project-MONAI/tutorials
"""
#Prostate MRI dataloader with pre-processing of full MRI images centre cropped to (192, 192, 96)
class ProstateMRIDataset(Dataset):
    def __init__(self, csv_file_img, binarise, datatype):
        self.data = pd.read_csv(csv_file_img)
        self.datatype = datatype
        self.binarise = binarise
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
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size= (192, 192, 96),
                ),

                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 96),
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
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 96),
                ),
                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 96),
                ),
    

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
        label = torch.where(label > 0, 1, 0) if self.binarise == True else label
        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        if self.datatype == 'train':
            sample = self.train_transforms(sample)
        else:
            sample = self.val_transforms(sample)

        return sample


class ProstateMRIDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, binarise, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size =batch_size
        self.num_workers = num_workers


        self.train_set = ProstateMRIDataset(self.csv_train_img, binarise, datatype='train')
        self.val_set = ProstateMRIDataset(self.csv_val_img, binarise, datatype= 'val')
        self.test_set = ProstateMRIDataset(self.csv_test_img, binarise, datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

#Abdominal CT dataloader with preprocessing into patches of size (96, 96, 96)
class AbdominalCTDataset(Dataset):
    def __init__(self, csv_file_img, binarise, datatype):
        self.data = pd.read_csv(csv_file_img)
        self.datatype = datatype
        self.binarise = binarise
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
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
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
        label = torch.where(label > 0, 1, 0) if self.binarise == True else label
      
        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        if self.datatype == 'train':
            sample = self.train_transforms(sample)
        else:
            sample = self.val_transforms(sample)

        return sample

class AbdominalCTDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, binarise, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size =batch_size
        self.num_workers = num_workers


        self.train_set = AbdominalCTDataset(self.csv_train_img, binarise, datatype='train')
        self.val_set = AbdominalCTDataset(self.csv_val_img, binarise, datatype= 'val')
        self.test_set = AbdominalCTDataset(self.csv_test_img, binarise, datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)
# Chest x-ray dataloader with preprocessing of images into size (256, 256)

class ChestxrayDataset(Dataset):
    def __init__(self, csv_file_img, binarise, datatype, pseudo_rgb =False):
        self.pseudo_rgb = pseudo_rgb
        self.data = pd.read_csv(csv_file_img)
        self.preprocess = T.Compose([
            T.Resize(256, 256),
            T.equalize(),
        ])

        self.normalise = lambda images: images * (2/255) - 1

        self.augment = T.RandomChoice([
            T.RandomHorizontalFlip(),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))]),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = self.data.loc[idx, 'images']
            img_label = self.data.loc[idx, 'labels']

            sample = {'image': img_path, 'label': img_label}

            self.samples.append(sample)

    def prepare_label(self, label_path):
        mask = Image.open(label_path)
        mask = mask.resize((256, 256))
        mask = np.array(mask)
        mask = ((np.array(mask) / 255) > .5).astype(int)
        # separate masks
        mask = np.swapaxes(mask, 0, 1)
        mask = measure.label(mask)
        mask = np.swapaxes(mask, 1, 0)
        return mask

    def preprocess_image(self, image):
        return self.normalise(T.equalize(image))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0).float()
        label = torch.from_numpy(sample['label']).float()

        image = self.augment(image)
        label = self.augment(label)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = Image.open(sample['image'])
        image = np.array(Image.resize((256, 256)))
        label = self.prepare_label(sample['label'])

        return {'image': image, 'label': label}


class ChestXrayDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img,  batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = ChestxrayDataset(self.csv_train_img, binarise, datatype='train')
        self.val_set = ChestxrayDataset(self.csv_val_img, binarise, datatype='val')
        self.test_set = ChestxrayDataset(self.csv_test_img, binarise, datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

