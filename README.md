# Vector-Quantisation-for-Robust-Segmentation
This is a repo for our work on Vector-Quantisation-for-Robust-Segmentation

## Description

This repo contains the code for training models with an autoencoder structure with a Vector Quantisation Blocks inserted into the bottleneck.
This codebase contains training and model code for the VQ-UNet and VQ-transUNet.
## Getting Started

### Dependencies

* Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### Datasets

* You will need to create 3 csv files (train.csv, validation.csv, test.csv) with two colums ('images', 'labels') containing the paths to the images and 
corresponding segmentations. We support nifti and png format. We provide an example for Prostate data in data/Prostate.
* You are free to choose to train on the dataset of your choice, pre-processed as you wish. We have provide dataloaders for the following 3 datasets.
  * * Abdomen: We use the Beyond the Cranial Vault (BTCV) dataset downloaded from the [Synape](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
  * * Chest: We use the [NIH](https://nihcc.app.box.com/s/r8kf5xcthjvvvf6r7l1an99e1nj4080m) chest x-ray dataset and the [JRST](http://db.jsrt.or.jp/eng.php) chest x-ray dataset.
  *.* Prostate: The prostate dataset is acquired from the [NCI-ISBI13](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) 
Challenge.

### Training/Testing.
* You can run the training/testing script together with main.py. You must enter the paths to the train, validation and test csv files and the output 
directory to save results and images. You will need to adjust other hyper-parameters according to your dataset which can be seen in main.py. 
Below is an example for Prostate data
```
python main.py --training_data '.../VQRobustSegmentation/data/Prostate/train.csv' --validation_data '.../VQRobustSegmentation/data/Prostate/validation.csv' --test_data '.../VQRobustSegmentation/data/Prostate/test.csv', --output_directory '.../VQRobustSegmentation/data/Prostate/output/'
```

## Authors

Contributors names and contact info

Ainkaran Santhirasekaram (a.santhirasekaram19@imperial.ac.uk)
Avinash Kori (a.kori21@imperial.ac.uk)


## References

* [Trans-UNet](https://github.com/Beckschen/TransUNet)
* [Taming-Transformers](https://github.com/CompVis/taming-transformers)

