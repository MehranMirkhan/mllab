
from sacred import Experiment
from tqdm import tqdm
import numpy as np
import visdom
import time
import gym


ex = Experiment('Q-Learning-01')
vis = visdom.Visdom()


# -----------  SYSTEM  -----------

class RLSys(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, steps=1, callback=None):
        observation = self.env.reset()
        loop = tqdm(range(steps))
        for step in loop:
            action = self.agent.act(observation)
            new_observation, reward, done, info = self.env.step(action)
            self.agent.learn(step, observation, action,
                             new_observation, reward, done)
            if callback is not None:
                callback(loop, step, observation, action,
                         new_observation, reward, done, info)
            observation = new_observation
            if done:
                observation = self.env.reset()
        loop.close()
        self.env.close()

    def test(self, steps=1, callback=None):
        observation = self.env.reset()
        loop = tqdm(range(steps))
        for step in loop:
            action = self.agent.act(observation)
            new_observation, reward, done, info = self.env.step(action)
            if callback is not None:
                callback(loop, step, observation, action,
                         new_observation, reward, done, info)
            observation = new_observation
            if done:
                observation = self.env.reset()
        loop.close()
        self.env.close()


# -----------  ENVIRONMENT  -----------

class FrozenLakeWrapper(object):
    def __init__(self):
        self.env = gym.make("FrozenLake-v0", is_slippery=False)
        # self.env = gym.make("FrozenLake8x8-v0", is_slippery=False)
        self.success_counter = 0

    def render(self, observation, scale=32):
        world = self.env.render(mode='ansi')
        # Removing color codes
        world = list(filter(lambda x: x in 'SFHG', world))
        world[observation] = 'A'
        mapper = {
            'S': [0, 0, 1],         # Start
            'F': [1, 1, 1],         # Frozen
            'H': [0, 0, 0],         # Hole
            'G': [0, 1, 0],         # Goal
            'A': [1, 0, 0],         # Agent
        }
        world = list(map(lambda x: mapper[x], world))
        world = np.array(world, dtype=np.float32)
        world = world.transpose().reshape(3, 4, 4)
        world = np.repeat(world, scale, axis=1)
        world = np.repeat(world, scale, axis=2)
        return world

    @ex.capture
    def train_callback(self, loop, step, observation, action,
                       new_observation, reward, done, info,
                       delay, render_interval):
        world = self.render(new_observation)
        if step % render_interval == 0:
            vis.image(world, win='lake')
        if done and reward > 0:
            self.success_counter += 1
            loop.set_description(desc=f"Successes = {self.success_counter}")
        time.sleep(delay)

    @ex.capture
    def test_callback(self, loop, step, observation, action,
                      new_observation, reward, done, info,
                      delay, render_interval):
        world = self.render(new_observation)
        vis.image(world, win='lake')
        # tqdm.write(f"{'LDRU'[action]}")
        # if done and reward > 0:
        #     tqdm.write("success")
        time.sleep(0.25)


# -----------  AGENT  -----------

class Agent(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):
        raise NotImplementedError

    def learn(self, step, prev_observation, action,
              new_observation, reward, done):
        pass


class RandomAgent(Agent):
    def act(self, observation):
        return self.action_space.sample()


class QAgent(Agent):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.n_state = observation_space.n
        self.n_act = action_space.n
        # self.q_table = np.random.randn(self.n_state, self.n_act) / 10
        self.q_table = np.zeros([self.n_state, self.n_act])

    def getQ(self, state, action):
        return self.q_table[state, action]

    def getStateQ(self, state):
        return self.q_table[state, :]

    def act(self, observation):
        stateQ = self.getStateQ(observation)
        probs = softmax(stateQ)
        action = np.random.choice(self.n_act, 1, p=probs)
        return action[0]
        # return np.argmax(stateQ)

    @ex.capture
    def learn(self, step, prev_observation, action,
              new_observation, reward, done,
              alpha, gamma, render_interval):
        # Calculating expected future reward
        futureQs = self.getStateQ(new_observation)
        expectedQ = np.max(futureQs)

        # Updating current Q-value
        self.q_table[prev_observation, action] = \
            (1 - alpha) * self.q_table[prev_observation, action] + \
            alpha * (reward + gamma * expectedQ)

        # Updating Q-value of final state
        if done:
            self.q_table[new_observation, :] = reward * 2 - 1

        # Displaying Q-Table
        if step % render_interval == 0:
            vis.heatmap(self.q_table, win='q-table')
            best_acts = np.argmax(self.q_table, axis=1)
            for i in range(self.n_state):
                if self.q_table[i, 0] in [-1, 1]:
                    best_acts[i] = -1
            mapper = {
                -1: blank_square,
                0: left_arrow,
                1: down_arrow,
                2: right_arrow,
                3: up_arrow,
            }
            best_acts = list(map(lambda x: mapper[x], best_acts))
            best_acts = np.array(best_acts, dtype=np.float32)
            best_acts = best_acts.reshape(4, 4, 8, 8, 3)
            best_acts = np.transpose(best_acts, [4, 0, 2, 1, 3])
            best_acts = best_acts.reshape(3, 32, 32)
            scale = 8
            best_acts = np.repeat(best_acts, scale, axis=1)
            best_acts = np.repeat(best_acts, scale, axis=2)
            vis.image(best_acts, win='actions')


# -----------  UTILS  -----------

def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec), axis=0)


B = [0, 0, 0]
W = [1, 1, 1]

left_arrow = [
    [W, W, W, W, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, B, B, W, W, W, W],
    [W, B, B, B, B, B, B, W],
    [W, W, B, B, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
]
right_arrow = [
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, B, W, W, W],
    [W, W, W, W, B, B, W, W],
    [W, B, B, B, B, B, B, W],
    [W, W, W, W, B, B, W, W],
    [W, W, W, W, B, W, W, W],
    [W, W, W, W, W, W, W, W],
    [W, W, W, W, W, W, W, W],
]
down_arrow = [
    [W, W, W, W, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, B, B, B, B, B, W, W],
    [W, W, B, B, B, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, W, W, W, W, W],
]
up_arrow = [
    [W, W, W, W, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, B, B, B, W, W, W],
    [W, B, B, B, B, B, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, B, W, W, W, W],
    [W, W, W, W, W, W, W, W],
]
blank_square = np.zeros([8, 8, 3])


# -----------  MAIN  -----------

@ex.config
def config():
    steps = int(1e4)
    delay = 0.001
    render_interval = 50
    alpha = 0.1
    gamma = 0.9


@ex.command
def test():
    print(softmax([3.0, 1.0, 0.2]))
    print([0.8360188, 0.11314284, 0.05083836])
    print(np.random.choice(3, 10, p=[0.8, 0.1, 0.1]))
    a = np.random.randn(2, 3)
    a[0, :] = 0
    print(a)


@ex.automain
def main(steps):
    flw = FrozenLakeWrapper()
    agent = QAgent(flw.env.observation_space, flw.env.action_space)
    rlsys = RLSys(flw.env, agent)
    rlsys.train(steps, flw.train_callback)
    rlsys.test(60, flw.test_callback)
