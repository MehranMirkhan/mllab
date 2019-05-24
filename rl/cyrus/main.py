import gym
from ex import ex
from agent import Physics


@ex.command
def hill_car_info():
    env = gym.make('MountainCar-v0')
    print(f"Obs: {env.observation_space}")
    print(f"Act: {env.action_space}")
    print(f"Obs: {env.observation_space.shape}")
    print(f"Act: {(env.action_space.n,)}")
    # Observation = (position, velocity)
    # Action = {0: left, 1: stay, 2: right}


@ex.command
def train_physics():
    env = gym.make('MountainCar-v0')
    agent = Physics(env)
    agent.train(1000)


@ex.automain
def main():
    comms = ex.gather_commands()
    print("Commands:")
    for c in comms:
        print(f"\t{c[0]}")
