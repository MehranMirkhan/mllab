
from sacred import Experiment
import numpy as np
import gym
import time
import visdom

ex = Experiment('GYM-LAKE')
vis = visdom.Visdom()


def render(world, observation):
    world = list(filter(lambda x: x in 'SFHG' , world))
    world[observation] = 'P'
    mapper = {
        'S': [0, 0, 1],
        'F': [1, 1, 1],
        'H': [0, 0, 0],
        'G': [0, 1, 0],
        'P': [1, 0, 0],
    }
    world = list(map(lambda x: mapper[x], world))
    world = np.array(world, dtype=np.float32)
    world = world.transpose().reshape(3, 8, 8)
    scale = 32
    world = np.repeat(world, scale, axis=1)
    world = np.repeat(world, scale, axis=2)
    vis.image(world, win='lake')


@ex.automain
def main():
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)
    observation = env.reset()
    for _ in range(100):
        world = env.render(mode='ansi')
        render(world, observation)
        env.render()
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        time.sleep(.1)

        if done:
            observation = env.reset()
    env.close()
