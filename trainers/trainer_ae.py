import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
)
from argparse import ArgumentParser
import time

import torch
import pytorch_lightning as pl

from models.AE import AE
from models.modules.autoencoder import AutoEncoder

# define hyper parameters
parser = ArgumentParser(description= 'Pytorch implementation of AutoEncoder')
# parser.add_argument('--model', type=str, default='AutoEncoder')
parser.add_argument('--input_dim', type=int, nargs='*', default=(1,28,28))
parser.add_argument('--encoder_conv_filters', type=int, nargs='*', default=[32,64,64,64])
parser.add_argument('--encoder_conv_kernel_size', type=int, nargs='*', default=[3,3,3,3])
parser.add_argument('--encoder_conv_strides', type=int, nargs='*', default=[1,1,1,1])
parser.add_argument('--decoder_conv_t_filters', type=int, nargs='*', default=[64,64,32,1])
parser.add_argument('--decoder_conv_t_kernel_size', type=int, nargs='*', default=[3,3,3,3])
parser.add_argument('--decoder_conv_t_strides', type=int, nargs='*', default=[1,1,1,1])
parser.add_argument('--z_dim', type=int, default=2)
# parser.add_argument('--use_batch_norm', type=bool, default=False)
parser.add_argument('--use_batch_norm', action='store_true')
# parser.add_argument('--use_dropout', type=bool, default=False)
parser.add_argument('--use_dropout', action='store_true')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=.0005)

parser.add_argument('--accelerator', type=str, default='cpu')
parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
# parser.add_argument('--log_every_n_steps', type=int, default=1000)

hparams = parser.parse_args()

# print('use_batch_norm : ', hparams.use_batch_norm)

# main
pl.seed_everything(1991)

callback_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    save_last = True,
    save_top_k = 4,
    # every_n_train_steps = 1000,
    # train_time_interval (Optional[timedelta]),
)

trainer = pl.Trainer.from_argparse_args(
    hparams,
    # fast_dev_run=False,
    # profiler= 'simple',
    # devices=3,
    # auto_lr_find = True
    callbacks = [callback_checkpoint],
    )
model = AE(hparams=hparams)

time_start = time.time()
trainer.fit(    
    model = model,
)
time_end = time.time()

print(f'elapsed time : {time_end - time_start}')