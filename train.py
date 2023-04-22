import os
from argparse import ArgumentParser
import time

import torch
import lightning.pytorch as pl

from models import AE
from datasets import mnist_dataset

# define hyper parameters
parser = ArgumentParser(description= 'Pytorch implementation of AutoEncoder')
parser.add_argument('--input_dim', type=int, nargs='*', default=(1,28,28))
parser.add_argument('--encoder_conv_filters', type=int, nargs='*', default=[32,64,64,64])
parser.add_argument('--encoder_conv_kernel_size', type=int, nargs='*', default=[3,3,3,3])
parser.add_argument('--encoder_conv_strides', type=int, nargs='*', default=[1,1,1,1])
parser.add_argument('--decoder_conv_t_filters', type=int, nargs='*', default=[64,64,32,1])
parser.add_argument('--decoder_conv_t_kernel_size', type=int, nargs='*', default=[3,3,3,3])
parser.add_argument('--decoder_conv_t_strides', type=int, nargs='*', default=[1,1,1,1])
parser.add_argument('--z_dim', type=int, default=2)
parser.add_argument('--use_batch_norm', action='store_true')
parser.add_argument('--use_dropout', action='store_true')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=.0005)

parser.add_argument('--accelerator', type=str, default='cpu')
parser.add_argument('--check_val_every_n_epoch', type=int, default=2)

# hparams = vars(parser.parse_args())
hparams = parser.parse_args()
# main
pl.seed_everything(1991)

model = AE.AE(
    hparams
)

callback_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode = 'min',
    save_last = True,
    save_top_k = 4,
    )

trainer = pl.Trainer(
    max_epochs= hparams.max_epochs,
    accelerator= hparams.accelerator,
    # devices=3,
    # val_check_interval= .4,
    check_val_every_n_epoch= 2,
    callbacks = [callback_checkpoint],
    # fast_dev_run=False,
    # precision= 16,
    # profiler= 'simple',
)
# lr_finder = trainer.tuner.lr_find(model, loader_train)
# model.hparams.lr = lr_finder.suggestion()

dataset_train= mnist_dataset.get_mnist_dataset(phase= 'train')
dataset_test= mnist_dataset.get_mnist_dataset(phase= 'test')

loader_train= torch.utils.data.DataLoader(dataset_train, batch_size= hparams.batch_size, shuffle=True)
loader_test= torch.utils.data.DataLoader(dataset_test, batch_size= hparams.batch_size, shuffle=False)

time_start = time.time()

trainer.fit(
    model = model,
    train_dataloaders= loader_train,
    val_dataloaders= loader_test,
)

time_end = time.time()

print(f'elapsed time : {time_end - time_start}')