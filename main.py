import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import VQUNet, VQtransUNet
from dataloaders import ProstateMRIDataModule, AbdominalCTDataModule, ChestXrayDataModule
from trainer import Train

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def none_or_int(value):
    if value == 'None':
        return None
    return value

def get_args_parser():
    parser = argparse.ArgumentParser('Vector Quantisation  training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='VQUnet', type=str,
                        help='Name of model to train')
    parser.add_argument('--image_size',  nargs="+",type=int, default=[192, 192, 64],
                        help='Size of input into the model. Prostate: [192, 192, 64], abdomen: [96, 96, 96], chest xray: [256, 256]')
    parser.add_argument('--dim', default='3D', type=str,
                        help='Dimension of image input')
    parser.add_argument('--drop_conv', type=float, default=0.0, 
                        help='Dropout rate for convolutional blocks (default: 0.)')
    parser.add_argument('--groups', type=int,  default=8, 
                        help='The number of groups for grouped normalisation. If None then batch normalisation')
    parser.add_argument('--in_ch', type=int,  default=1,
                        help='The number of input channels')
    parser.add_argument('--channels', type=int,  default=8,
                        help='The number of channels from first level of encoder')
    parser.add_argument('--enc_blocks' , nargs="+",type=int, default=[1, 1, 1, 1],
                        help='Number of ResBlocks per level of the encoder')
    parser.add_argument('--dec_blocks', nargs="+",type=int, default=[1, 1, 1],
                        help='Number of ResBlocks per level of the decoder')
    parser.add_argument('--act', default='swish', type=str,
                        help='Activation function to use. Enter "swish" if swish activation or enter function in torch function format if required different activation i.e. "nn.ReLU()"')
    parser.add_argument('--with_conv', type=str_to_bool, default=False,
                        help='Applying upsampling with convolution')
    parser.add_argument('--VQ', type=str_to_bool, default=True,
                        help='Apply vector quantisation in the bottleneck of the architecture. If False, then turns into original architecture')
    parser.add_argument('--pos_encoding', type=str_to_bool, default=False,
                        help='Apply Positional encoding before VQ in the bottleneck of the architecture.')
    parser.add_argument('--quantise', default='spatial', type=str, choices=['spatial', 'channel'],
                        help='Quantise either spatially or channel wise, enter either "spatial" or "channel"')
    parser.add_argument('--n_e', default=1024, type=int,
                        help='The number of codebook vectors')
    parser.add_argument('--embed_dim', default=256, type=int,
                        help='The number of channels before quantisation. If quantising spatially, this will be equivelent to the codebook dimension')
    
    #Transformer parameter if using transunet
    parser.add_argument('--trans_attn', default='spatial', type=str, choices=['spatial', 'channel'],
                        help='Perform attention either spatially or channel wise, enter either "spatial" or "channel"')
    parser.add_argument('--trans_layers', default=8, type=int,
                        help='The number of transformer layers')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='The number of attention_heads')
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help='The hidden layer dimension size in the MLP layer of the transformer')
    parser.add_argument('--drop_attn', type=float, default=0.2,
                        help='Dropout rate for attention blocks (default: 0.2)')
    # Dataset parameters
    parser.add_argument('--dataset', default='prostate', type=str, choices=['prostate', 'abdomen', 'chest'],
                        help='Pick dataset to use, this can be extended if the user wishes to create their own dataloaders.py file')
    parser.add_argument('--training_data', default='.../VQRobustSegmentation/data/Prostate/train.csv', type=str, required = True,  
                        help='training data csv file')
    parser.add_argument('--validation_data', default='.../VQRobustSegmentation/data/Prostate/validation.csv', type=str, required = True,
                        help='validation data csv file')
    parser.add_argument('--test_data', default='.../VQRobustSegmentation/data/Prostate/test.csv', type=str, required = True,
                        help='test data csv file')
    parser.add_argument('--binarise', type=str_to_bool, default=True,
                        help='Choose whether to binarise the segmentation map')
    parser.add_argument('--image_format', default='nifti',choices=['nifti', 'png'], type=str, 
                        help='state if image in nifti or png format')
    parser.add_argument('--labels', "--list", nargs="+",default=['Whole Prostate'],  
                        help='Label of classes. If not binarise, Prostate:["TZ", "PZ"], Chest: ["L-lung", "R-lung"], Abdomen: ["spleen",  "rkid", "lkid",  "gall", "eso", "liver", "sto",  "aorta",  "IVC",  "veins", "pancreas",  "rad", "lad"]')
    
    # Training parameters
    parser.add_argument('--classes', type=int, default=3, metavar='LR',
                        help='number of classes. 14 for abdomen CT, 3 for Prostate MRI, 3 for chest x-ray.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='The number of epochs after which to evaluate the validation set')
    parser.add_argument('--sliding_inference', type=str_to_bool, default=False,
                        help='Choose whether to perform sliding window inference. Only required for abdomen')
    
    #GPU usage
    parser.add_argument('--gpus', type=int, default=1, 
                        help='number oF GPUs')
    parser.add_argument('--nodes', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether to use deterministic training')

    #Output directory
    parser.add_argument('--output_directory', default='.../VQRobustSegmentation/data/Prostate/output/', type=str, required = True, 
                        help='output directory to save images and results')
    return parser

def main(args):
    pl.seed_everything(args.seed, workers=True)
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    if 'tran' in args.model:
        model = VQtransUNet(trans_layers = args.trans_layers, image_size = args.image_size, hidden_dim = args.hidden_dim, num_heads = args.num_heads, 
                channels = args.channels ,groups = args.groups, in_ch = args.in_ch, out_ch = 2 if args.binarise else args.classes, enc_blocks = args.enc_blocks, dec_blocks = args.dec_blocks, dim = args.dim, 
                act= args.act,  with_conv = args.with_conv, embed_dim = args.embed_dim, n_e = args.n_e,  quantise = args.quantise, VQ = args.VQ, 
                pos_encoding = args.pos_encoding, trans_attn=args.trans_attn, conv_dropout=args.drop_conv, trans_dropout = args.drop_attn)
    else:
        model = VQUNet(image_size = args.image_size,channels = args.channels, in_ch = args.in_ch, out_ch = 2 if args.binarise else args.classes, groups = args.groups, enc_blocks = args.enc_blocks, 
                dec_blocks = args.dec_blocks, dim = args.dim, act= args.act,  with_conv = args.with_conv, embed_dim = args.embed_dim, n_e = args.n_e,  
                quantise = args.quantise, VQ = args.VQ, pos_encoding = args.pos_encoding,  dropout=args.drop_conv)

    if args.dataset == 'prostate':
        data = ProstateMRIDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binarise = args.binarise,
            csv_train_img=args.training_data,
            csv_val_img=args.validation_data,
            csv_test_img=args.test_data,
        )
    elif args.dataset == 'abdomen':
        data = AbdominalCTDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binarise = args.binarise,
            csv_train_img=args.training_data,
            csv_val_img=args.validation_data,
            csv_test_img=args.test_data,
        )
    elif args.dataset == 'chest':
        data = ChestXrayDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binarise = args.binarise,
            csv_train_img=args.training_data,
            csv_val_img=args.validation_data,
            csv_test_img=args.test_data,
        )
    out_dir = args.output_directory
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok = True)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok = True)
    net = Train(model = model, weight_decay = args.weight_decay, lr = args.lr,
            sliding_inference = args.sliding_inference, ROI = args.image_size, output_dir = out_dir,
            image_format = args.image_format, labels = args.labels, num_classes = 2 if args.binarise else args.classes)
    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(verbose=True,
        monitor="mean_val_loss",
        save_top_k = 1,
        every_n_epochs = args.val_epoch,
        mode="min",
        dirpath=out_dir, 
        )
    # initialise Lightning's trainer.
    trainer = pl.Trainer(
              gpus=args.gpus, accelerator='ddp', num_nodes=args.nodes,
              max_epochs=args.epochs,
              check_val_every_n_epoch = args.val_epoch,
              callbacks=checkpoint_callback,
    )
    trainer.fit(net, data)
    model = net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model = model, weight_decay = args.weight_decay, lr = args.lr,
            sliding_inference = args.sliding_inference, ROI = args.image_size, output_dir = out_dir,
            image_format = args.image_format, labels = args.labels, num_classes = 2 if args.binarise else args.classes)
    model.eval()
    trainer.test(model, data.test_dataloader())    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vector quantisation for segmentation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if 'tran' in args.model:
        del args.trans_layers, args.hidden_dim, args.num_heads, args.trans_attn, args.trans_dropout
    if args.dataset == 'chest' or 'prostate':
        args.sliding_inference == False    
    main(args)
