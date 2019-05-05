from sacred import Experiment
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

ex = Experiment('pytorch-base')

#-------------------------- CONFIG
@ex.config
def config():
    arch = [28*28, 48, 10]
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    learning_rate = 1e-3
    dataset_root = './data'
    batch_size = 64
    epochs = 3

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#-------------------------- MODEL
class Model(torch.nn.Module):
    @ex.capture
    def __init__(self, arch):
        super().__init__()
        self.l1 = torch.nn.Linear(arch[0], arch[1])
        self.l2 = torch.nn.Linear(arch[1], arch[2])

    def forward(self, x):
        h = torch.relu(self.l1(x))
        o = torch.sigmoid(self.l2(h))
        return o

    def learn(self, x, y, criterion, optimizer):
        o = self.forward(x)
        loss = criterion(o, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

#-------------------------- DATASET
class Dataset(object):
    @ex.capture
    def __init__(self, dataset_root, batch_size):
        trans = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,),
                                                        (0.5,))])
        train_set = dset.MNIST(root=dataset_root, train=True,
                               transform=trans, download=True)
        test_set = dset.MNIST(root=dataset_root, train=False,
                              transform=trans, download=True)
        loader = torch.utils.data.DataLoader
        self.train_loader = loader(dataset=train_set,
                                   batch_size=batch_size,
                                   shuffle=True)
        self.test_loader = loader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=False)

#-------------------------- TEACHER
class Teacher(object):
    @ex.capture
    def __init__(self, model, dataset, criterion, optimizer, learning_rate):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)

    @ex.capture
    def train(self, epochs):
        for epoch in range(epochs):
            loop = tqdm(self.dataset.train_loader)
            for x, y in loop:
                x = x.view(-1, 28*28)
                loss = self.model.learn(x, y, self.criterion, self.optimizer)
                loop.set_description(f"loss={loss:.5f}")
            loop.close()
            self.test()

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.dataset.test_loader:
                x = x.view(-1, 28*28)
                o = self.model.forward(x)
                _, pred = torch.max(o.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        print(f"Accuracy = {100 * correct / total} %")

#-------------------------- MAIN
@ex.command
def show_dset():
    dataset = Dataset()
    dataiter = iter(dataset.train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

@ex.automain
def run():
    model = Model()
    dataset = Dataset()
    teacher = Teacher(model, dataset)
    teacher.train()
