
import torch

from experiment import ex


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


@ex.capture
def make_generator(Dz, Dh):
    arch = [Dz, Dh, 28*28]
    return torch.nn.Sequential(
        torch.nn.Linear(arch[0], arch[1]),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(arch[1], arch[1]),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(arch[1], arch[2]),
        torch.nn.Tanh(),
    )


@ex.capture
def make_discriminator(Dh):
    arch = [28*28, Dh, 1]
    return torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(arch[0], arch[1]),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(arch[1], arch[1]),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(arch[1], arch[2]),
        torch.nn.Sigmoid(),
    )


class MLP_GAN(object):
    @ex.capture
    def __init__(self, learning_rate):
        self.generator = make_generator()
        self.discriminator = make_discriminator()
        self.criterion = torch.nn.BCELoss()
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate)
        self.dis_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate)

    @ex.capture
    def generate(self, batch_size, Dz):
        z = torch.randn(batch_size, Dz)
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def learn(self, x):
        batch_size = x.size()[0]
        ones = torch.ones(batch_size, 1, requires_grad=False)

        # Train generator
        self.gen_optimizer.zero_grad()
        fake = self.generate(batch_size=batch_size)
        o_fake = self.discriminate(fake)
        gen_loss = self.criterion(o_fake, ones)
        gen_loss.backward()
        self.gen_optimizer.step()

        # Train discriminator
        self.dis_optimizer.zero_grad()
        o_real = self.discriminate(x)
        fake = self.generate(batch_size=batch_size)
        o_fake = self.discriminate(fake)
        dis_loss = (self.criterion(o_real, ones) +
                    self.criterion(o_fake, ones - 1)) / 2
        dis_loss.backward()
        self.dis_optimizer.step()

        return dis_loss.item(), gen_loss.item()

    def save(self, model_path):
        d = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }
        torch.save(d, model_path)
    
    def load(self, model_path):
        d = torch.load(model_path)
        self.generator.load_state_dict(d['generator'])
        self.discriminator.load_state_dict(d['discriminator'])
        self.generator.eval()
        self.discriminator.eval()
