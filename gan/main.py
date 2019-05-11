
from tqdm import tqdm
import visdom
import numpy as np

from experiment import ex
from dataset import MNIST
from model import MLP_GAN
from teacher import UnsupervisedTeacher

vis = visdom.Visdom()


@ex.capture
def batch_callback(model, loss_width):
    counter = 0
    y = []
    vis.line(X=[0], Y=[[0, 0]], win='loss',
             opts={'legend': ['dis', 'gen']})

    def callback(epoch, batch_index, loss):
        nonlocal counter
        update = 'replace' if epoch == 0 and batch_index == 0 else 'append'
        if batch_index % 20 == 0:
            sample = model.generate().detach()
            sample = sample.numpy().reshape((-1, 1, 28, 28))
            sample = sample * 0.5 + 0.5
            scale = 2
            sample = np.repeat(sample, scale, axis=2)
            sample = np.repeat(sample, scale, axis=3)
            vis.images(sample, win='gen')
        y.append(list(loss))
        if len(y) > loss_width:
            y.pop(0)
        x = list(range(counter - len(y) + 1, counter + 1))
        vis.line(X=x, Y=y,
                 update='replace', win='loss',
                 opts={'legend': ['dis', 'gen']})
        counter += 1
    return callback


@ex.capture
def epoch_callback(model, save_model, model_path):
    def callback(epoch):
        if save_model:
            tqdm.write('Saving model...')
            model.save(model_path)
            tqdm.write('...Done.')
    return callback


@ex.command
def train(load_model, model_path):
    model = MLP_GAN()
    if load_model:
        print('Loading model...')
        model.load(model_path)
        print('...Done.')
    ds = MNIST()
    teacher = UnsupervisedTeacher(model, ds)
    if vis.check_connection():
        teacher.train(epoch_callback=epoch_callback(model),
                      batch_callback=batch_callback(model))
    else:
        teacher.train(epoch_callback=epoch_callback(model))


@ex.automain
def main():
    print("""
    Commands:
        train
    """)
