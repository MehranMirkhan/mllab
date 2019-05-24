from sacred import Experiment

ex = Experiment('CYRUS')


@ex.config
def config():
    physics_lr = 1e-3
