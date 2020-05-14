import numpy as np
import matplotlib.pylab as plt
from renderer import *
from datetime import datetime
import os


class Agent:
    def __init__(self, i, world):
        self.i = i
        self.world = world
        self.x = np.random.uniform(size=2)
        self.v = np.zeros_like(self.x)
        self.history = []
        self.store_history()

    def update(self, action):
        self.v += action

        norm = np.linalg.norm(self.v)
        if norm > .1:
            self.v /= norm
            self.v *= .1
        self.x += self.v

        if np.max(np.abs(self.x)) > 2.:
            self.v = np.zeros_like(self.v)
        self.x = np.clip(self.x, -1.95, 1.95)

        self.store_history()

    def get_state(self):
        return np.concatenate((self.x, self.v)).flatten()

    def store_history(self):
        self.history.append(np.copy(self.x))


class World:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()
        self.state_dim = 8
        self.action_dim = 2

    def reset(self):
        self.agents = [Agent(i, self) for i in range(self.n_agents)]
        return self.get_state()

    def get_state(self):
        return [agent.get_state() for agent in self.agents]

    def step(self, action):
        self.agents[0].update(np.random.uniform(-.1, .1, size=2))
        self.agents[1].update(action * .1)
        d = self.agents[0].x - self.agents[1].x
        d = np.linalg.norm(d)
        r = -d
        event = d < .1
        s = [agent.get_state() for agent in self.agents]
        return s, r, False, event


class DiscreteWorld(World):
    def __init__(self, n_agents):
        super().__init__(n_agents)
        self.action_dim = 5

    def transform_action(self, action):
        if action == 0:
            action = [0., 0.]
        elif action == 1:
            action = [.1, 0.]
        elif action == 2:
            action = [.0, .1]
        elif action == 3:
            action = [-.1, 0.]
        elif action == 4:
            action = [0., -.1]
        return action

    def step(self, action):
        action = self.transform_action(action)
        return super().step(action)


class TargetWorld(DiscreteWorld):
    def __init__(self, n_agents):
        super().__init__(n_agents)

    def step(self, action):
        action = self.transform_action(action)
        self.agents[0].update(np.random.uniform(-.1, .1, size=2))
        self.agents[1].update(action)
        d = self.agents[0].x - self.agents[1].x
        d = np.linalg.norm(d)
        r = -1
        event = d < .1
        s = [agent.get_state() for agent in self.agents]
        done = np.linalg.norm(self.agents[1].x) < .1
        return s, r, done, event


class MoveAgent:
    def __init__(self):
        self.x = np.random.randint(0, 5, size=2)
        self.history = []
        self.store_history()

    def update(self, action):
        self.x += action
        self.x = np.clip(self.x, 0, 5)
        self.store_history()

    def store_history(self):
        self.history.append(np.copy(self.x))


class MoveWorld:
    def __init__(self):
        super().__init__()
        self.reset()
        self.state_dim = 6
        self.actions = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
        self.action_dim = len(self.actions)

    def reset(self):
        self.agents = [MoveAgent() for _ in range(2)]
        self.target = np.zeros(2)
        return self.get_state()

    def step(self, action):
        self.agents[0].update(
            self.actions[np.random.randint(len(self.actions))])
        self.agents[1].update(self.transform_action(action))
        event = all(self.agents[0].x == self.agents[1].x)
        done = all(self.agents[1].x == self.target)
        return self.get_state(), -1, done, event

    def get_state(self):
        return np.concatenate((np.array([a.x for a in self.agents]).flatten(), self.target)).flatten()

    def transform_action(self, action):
        return self.actions[action]


class MoveAgentContinuous:
    def __init__(self, init_low, init_high):
        self.x = np.random.uniform(init_low, init_high, size=2)
        self.v = np.zeros_like(self.x)
        self.history = []
        self.store_history()

    def update(self, action):
        self.v += action
        norm = np.linalg.norm(self.v)
        if norm > 1:
            self.v /= norm
        self.x += self.v
        if max(self.x) > 1 or min(self.x) < -1:
            self.v = np.zeros_like(self.v)
        self.x = np.clip(self.x, -1, 1)
        self.store_history()

    def store_history(self):
        self.history.append(np.copy(self.x))


class MoveWorldContinuous:
    def __init__(self):
        super().__init__()
        self.reset()
        self.state_dim = 6
        self.action_dim = 2

    def reset(self):
        self.agents = [MoveAgentContinuous(0 - i, .5 - i) for i in range(2)]
        self.target = np.random.uniform(.5, 1, size=2)
        return self.get_state()

    def step(self, action):
        self.agents[0].update(np.random.uniform(-.05, .05, size=2))
        self.agents[1].update(action * .05)
        d = self.agents[0].x - self.agents[1].x
        d = np.linalg.norm(d)
        event = d < .1
        d_target = np.linalg.norm(self.target - self.agents[1].x)
        r = - 1
        done = d_target < .1
        return self.get_state(), r, done, event

    def get_state(self):
        return np.concatenate((np.array([a.x for a in self.agents]).flatten(), self.target)).flatten()

    def transform_action(self, action):
        return self.actions[action]
