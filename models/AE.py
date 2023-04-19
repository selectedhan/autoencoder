# import numpy as np
import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import torch
import torchvision
import lightning.pytorch as pl

from modules.autoencoder import AutoEncoder

class AE(pl.LightningModule):
    def __init__(self, **hparams) :
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.model = AutoEncoder(
            self.hparams.input_dim,
            self.hparams.encoder_conv_filters,
            self.hparams.encoder_conv_kernel_size,
            self.hparams.encoder_conv_strides,
            self.hparams.decoder_conv_t_filters,
            self.hparams.decoder_conv_t_kernel_size,
            self.hparams.decoder_conv_t_strides,
            self.hparams.z_dim,
            self.hparams.use_batch_norm,
            self.hparams.use_dropout,    
        )
        self.optimizer = torch.optim.Adam
        self.criterion = torch.nn.MSELoss()

    def prepare_data(self):
        data_train = torchvision.datasets.MNIST(
            root = 'data',
            train=True,
            download=True,
            transform = torchvision.transforms.ToTensor()
        )

        self.dataset_train, self.dataset_val = torch.utils.data.random_split(data_train, [55000, 5000])
    
        self.dataset_test = torchvision.datasets.MNIST(
            root = 'data',
            train=False,
            download=True,
            transform = torchvision.transforms.ToTensor()
        )    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train, batch_size=self.hparams.batch_size)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_val, batch_size=self.hparams.batch_size)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_test, batch_size=self.hparams.batch_size)


    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X_reconstruction = self(X)

        loss = self.criterion(X_reconstruction, X)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        X, y = batch
        X_reconstruction = self(X)

        loss = self.criterion(X_reconstruction, X)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr = self.hparams.lr)

if __name__ == '__main__':
    import time

    BATCH_SIZE = 32
    EPOCH = 5
    LEARNING_RATE = .0005    

    model = AE(
        input_dim = (1,28,28),
        encoder_conv_filters = [1], #[32, 64, 64, 64],
        encoder_conv_kernel_size = [3], #[3, 3, 3, 3],
        encoder_conv_strides = [1], #[1, 1, 1, 1],    #[1, 2, 2, 1]
        decoder_conv_t_filters = [1], #[64, 64, 32, 1],
        decoder_conv_t_kernel_size = [3], #[3, 3, 3, 3],
        decoder_conv_t_strides = [1], #[1, 1, 1, 1],  # [1, 2, 2, 1]
        z_dim = 2,
        use_batch_norm = False,
        use_dropout = False,
        batch_size = BATCH_SIZE,
        lr = .05,
    )

    pl.seed_everything(1991)
    callback_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode = 'min',
        save_last = True,
        save_top_k = 4,
        )

    trainer = pl.Trainer(
        max_epochs= EPOCH,
        accelerator= 'cpu',
        # devices=3,
        val_check_interval= .4,
        callbacks = [callback_checkpoint],
        # fast_dev_run=False,
        # precision= 16,
        # profiler= 'simple',
    )
    # lr_finder = trainer.tuner.lr_find(model, loader_train)
    # model.hparams.lr = lr_finder.suggestion()

    time_start = time.time()
    trainer.fit(    
        model = model,
    )
    time_end = time.time()

    print(f'elapsed time : {time_end - time_start}')
    # devices =4 : runtime error / runtime error
    # devices =3 : runtime error / runtime error
    # devices =2 : 109.1679 / broken_pipe error
    # devices =1 : 119.8189 / 111.1125
    # devices ='auto' : 124.8250 / 102.8464
