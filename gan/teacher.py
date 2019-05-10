
from tqdm import tqdm

from experiment import ex


class SupervisedTeacher(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    @ex.capture
    def train(self, epochs, epoch_test=True, epoch_callback=None, batch_callback=None):
        for epoch in range(epochs):
            tqdm.write(f"Epoch {epoch}")
            batch_index = 0
            for x, y in tqdm(self.dataset.train_loader):
                loss = self.model.learn(x, y)
                if batch_callback is not None:
                    batch_callback(epoch, batch_index, loss)
                batch_index += 1
            if epoch_callback is not None:
                epoch_callback(epoch)
            if epoch_test:
                self.test()

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.dataset.test_loader:
                o = self.model.forward(x)
                _, pred = torch.max(o.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        return correct / total


class UnsupervisedTeacher(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    @ex.capture
    def train(self, epochs, epoch_callback=None, batch_callback=None):
        for epoch in range(epochs):
            tqdm.write(f"Epoch {epoch}")
            batch_index = 0
            for x, _ in tqdm(self.dataset.train_loader):
                loss = self.model.learn(x)
                if batch_callback is not None:
                    batch_callback(epoch, batch_index, loss)
                batch_index += 1
            if epoch_callback is not None:
                epoch_callback(epoch)
