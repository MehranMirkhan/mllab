
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from experiment import ex


class Dataset(object):
    @ex.capture
    def __init__(self, dataset, dataset_root, batch_size, mean, std):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
        train_set = dataset(root=dataset_root, train=True,
                            transform=trans, download=True)
        test_set = dataset(root=dataset_root, train=False,
                           transform=trans, download=True)
        loader = torch.utils.data.DataLoader
        self.train_loader = loader(dataset=train_set,
                                   batch_size=batch_size,
                                   shuffle=True)
        self.test_loader = loader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=False)


class MNIST(Dataset):
    def __init__(self):
        super().__init__(dataset=dset.MNIST)
