from sacred import Experiment
import numpy as np
import math
import time
import visdom

ex = Experiment('visdom-sandbox')

@ex.command
def image():
    vis = visdom.Visdom()
    vis.close(win='win1')
    for _ in range(100):
        vis.image(np.random.rand(3, 256, 256), win='win1')
        time.sleep(0.1)

@ex.command
def line():
    vis = visdom.Visdom()
    vis.close(win='win2')
    for x in range(1, 501):
        y = math.log(x) + (np.random.rand() - 0.5) * 0.5
        vis.line(X=[x], Y=[y], update='append', win='win2')
        time.sleep(0.01)

@ex.automain
def run():
    print("Hello")
