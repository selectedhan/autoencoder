import numpy as np
import torch
# from torchsummary import summary
from typing import List

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(-1, *self.shape)

class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: List,
        encoder_conv_filters,
        encoder_conv_kernel_size: List,
        encoder_conv_strides: List,
        decoder_conv_t_filters: List,
        decoder_conv_t_kernel_size: List,
        decoder_conv_t_strides: List,
        z_dim: int,
        use_batch_norm: bool = False,
        use_dropout: bool = False,
    ) -> torch.nn.Module :
        
        super().__init__()
        self.name = 'autoencoder'
        
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        
        self._build_encoder()
        self._build_decoder()

    def _get_out_shape(self, model, input_dim) :
        return model(torch.rand(1, *input_dim)).detach().shape[1:]
    
    def _build_encoder(self):
        # Encoder
        encoder = torch.nn.Sequential()
        for i in range(self.n_layers_encoder):
            shape_conv_input = self._get_out_shape(encoder, self.input_dim)
            encoder.append(
                torch.nn.Conv2d(
                    in_channels= shape_conv_input[0],
                    out_channels= self.encoder_conv_filters[i],
                    kernel_size= self.encoder_conv_kernel_size[i],
                    stride= self.encoder_conv_strides[i],
                    padding= 'same'
                )
            )
            encoder.append(torch.nn.LeakyReLU())
            if self.use_batch_norm:
                encoder.append(torch.nn.BatchNorm2d())
            if self.use_dropout:
                encoder.append(torch.nn.Dropout(p= .25))
                
        self.shape_before_flattening = self._get_out_shape(encoder, self.input_dim)
        
        encoder.append(torch.nn.Flatten())
        encoder.append(
            torch.nn.Linear(
            in_features= np.prod(self.shape_before_flattening),
            out_features= self.z_dim
        )
        )
        self.encoder = encoder

    def _build_decoder(self):
        # Decoder
        decoder = torch.nn.Sequential()
        decoder.append(
            torch.nn.Linear(
            in_features= self.z_dim,
            out_features=np.prod(self.shape_before_flattening)
        )
        )
        decoder.append(Reshape(self.shape_before_flattening))

        for i in range(self.n_layers_decoder):
            shape_conv_input = self._get_out_shape(decoder, [self.z_dim])
            decoder.append(
                torch.nn.ConvTranspose2d(
                in_channels= shape_conv_input[0],
                out_channels= self.decoder_conv_t_filters[i],
                kernel_size= self.decoder_conv_t_kernel_size[i],
                stride= self.decoder_conv_t_strides[i],
                padding= self.decoder_conv_t_kernel_size[i] // 2
                # padding= 'same'
            )
            )            
            if i < self.n_layers_decoder - 1:
                decoder.append(torch.nn.LeakyReLU())

                if self.use_batch_norm:
                    decoder.append(torch.nn.BatchNorm2d())
            
                if self.use_dropout:
                    decoder.append(torch.nn.Dropout(p= .25))
            else :
                decoder.append(torch.nn.Sigmoid())
        
        self.decoder = decoder

    def forward(self, x):   
        return self.decoder(self.encoder(x))
    
    def get_codes(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    import torchvision

    EPOCH = 10
    BATCH_SIZE = 32

    print('load test model...', end='')
    model = AutoEncoder(
        input_dim = (1,28,28),
        encoder_conv_filters = [4],  #[32, 64, 64, 64],
        encoder_conv_kernel_size = [3], #[3, 3, 3, 3],
        encoder_conv_strides = [1], #[1, 1, 1, 1],    #[1, 2, 2, 1]
        decoder_conv_t_filters = [1], #[64, 64, 32, 1],
        decoder_conv_t_kernel_size = [3], #[3, 3, 3, 3],
        decoder_conv_t_strides = [1], #[1, 1, 1, 1],  # [1, 2, 2, 1]
        z_dim = 2,
        use_batch_norm = False,
        use_dropout = False,
    )
    print('completely load model !')
    # print(summary(model, (1, 28, 28)))


    print('load dataloader...', end='')
    data_train = torchvision.datasets.MNIST(
        root = 'data',
        train=True,
        download=True,
        transform = torchvision.transforms.ToTensor()
    )

    dataset_train, dataset_val = torch.utils.data.random_split(data_train, [55000, 5000])
    
    dataset_test = torchvision.datasets.MNIST(
        root = 'data',
        train=False,
        download=True,
        transform = torchvision.transforms.ToTensor()
    )

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)
    print('completely load dataloader !')

    print('training start !')
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(EPOCH):
        epoch_loss = 0
        for idx, batch in enumerate(loader_train):
            X, y = batch
            X_reconstruction = model(X)

            loss = torch.nn.MSELoss()(X_reconstruction, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        print(f'[{epoch} / {EPOCH}]  training_loss: {epoch_loss / len(loader_train):.4f}')
            
