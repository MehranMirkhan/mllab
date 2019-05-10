
from sacred import Experiment

ex = Experiment('MLP-GAN')


@ex.config
def dataset():
    dataset_root = './data'
    batch_size = 64
    mean = (0.5,)
    std = (0.5,)


@ex.config
def model():
    Dz = 128
    Dh = 128
    generator_arch = []
    discriminator_arch = []
    epochs = 100
    learning_rate = 1e-4
    model_path = './gan/model.pickle'
    save_model = True
    load_model = True


@ex.config
def vis():
    loss_width = 500
