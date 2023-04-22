import torch
import torchvision

def get_mnist_dataset(phase, transform= torchvision.transforms.ToTensor()):
    dataset = torchvision.datasets.MNIST(
        root= 'data',
        train= (True if phase== 'train' else False),
        transform= transform,
        download= False,
    )
    return dataset

if __name__ == '__main__':
    dataset = get_mnist_dataset('test')
    sample= next(iter(dataset))
    print(f'len_dataset[0]: {len(sample)}')
    print(sample[0].shape)
    print(sample[1])