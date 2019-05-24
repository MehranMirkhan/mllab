from tqdm import tqdm
from ex import ex
import numpy as np
import torch
import time


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_shape = (env.action_space.n,)

    def act(self, observation):
        return self.env.action_space.sample()

    def learn(self, prev_observation, action,
              new_observation, reward, done):
        pass

    def reset(self):
        pass

    def train(self, steps=1):
        observation = self.env.reset()
        self.reset()
        loop = tqdm(range(steps))
        for step in loop:
            action = self.act(observation)
            new_observation, reward, done, info = self.env.step(action)
            self.learn(observation, action,
                       new_observation, reward, done)
            self.train_callback(loop, step, observation, action,
                                new_observation, reward, done, info)
            observation = new_observation
            if done:
                observation = self.env.reset()
                self.reset()
        loop.close()

    def train_callback(self, loop, step, observation, action,
                       new_observation, reward, done, info):
        pass


class Physics(Agent):
    @ex.capture
    def __init__(self, env, physics_lr):
        super().__init__(env)
        self.n_s = self.state_shape[0]
        self.n_a = self.action_shape[0]
        self.net = self.make_net(self.n_s + self.n_a, self.n_s)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=physics_lr)
        self.last_loss = 0

    def make_net(self, n_i, n_o):
        n_h = 32
        return torch.nn.Sequential(
            torch.nn.Linear(n_i, n_h),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(n_h, n_o),
        )

    def learn(self, prev_observation, action,
              new_observation, reward, done):
        self.optimizer.zero_grad()
        a = [0.0] * self.n_a
        a[action] = 1.0
        x = [list(prev_observation) + a]
        x = torch.Tensor(x)
        o = self.net(x)
        y = torch.Tensor([new_observation])
        loss = self.criterion(o, y)
        self.last_loss = np.log(loss.item())
        loss.backward()
        self.optimizer.step()

    def train_callback(self, loop, step, observation, action,
                       new_observation, reward, done, info):
        loop.set_description(f"{self.last_loss:.2f}")
        # self.env.render()
        # time.sleep(0.1)
