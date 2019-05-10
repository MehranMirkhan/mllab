from sacred import Experiment
import numpy as np
import math
import time
import visdom

ex = Experiment('visdom-sandbox')
vis = visdom.Visdom()


@ex.command
def image():
    win = 'image'
    for _ in range(300):
        vis.image(np.random.rand(3, 256, 256), win=win)
        time.sleep(0.01)


@ex.command
def line():
    win = 'line'
    for x in range(1, 301):
        update = 'replace' if x == 1 else 'append'
        y1 = math.log(x) + (np.random.rand() - 0.5) * 0.5
        y2 = math.log(x) / 2 + (np.random.rand() - 0.5) * 0.5
        vis.line(X=[x], Y=[[y1, y2]], update=update, win=win, opts=dict(
            legend=['y1', 'y2']))
        time.sleep(0.01)


@ex.automain
def run():
    print("Hello")
