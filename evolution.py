# based on https://github.com/BenSecret/Pytorch-Evolution-Strategies/blob/master/pytorch-evolution-strategies.py
import numpy as np
import torch
import torch.utils.data
from collections import deque
import parameters as params
from torch.distributions.normal import Normal

torch.manual_seed(params.seed)
np.random.seed(params.seed)


class EvolutionStrategies(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super(EvolutionStrategies, self).__init__()
        hidden = params.hidden_neurons
        self.linear1 = torch.nn.Linear(inputs, hidden)
        self.linear2 = torch.nn.Linear(hidden, outputs)
        self.population_size = params.population_size
        self.sigma = params.sigma
        self.learning_rate = params.learning_rate
        self.counter = 0
        self.rewards = []
        self.costs = []
        self.score_tracking = deque(maxlen=100)
        self.master_weights = []
        self.c_sat = 1.

        for param in self.parameters():
            self.master_weights.append(param.data)
        self.populate()

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return torch.tanh(self.linear2(x))

    def populate(self):
        self.population = []
        for _ in range(self.population_size):
            x = []
            for param in self.parameters():
                x.append(np.random.randn(*param.data.size()))
            self.population.append(x)

    def add_noise_to_weights(self):
        for i, param in enumerate(self.parameters()):
            noise = torch.from_numpy(
                self.sigma * self.population[self.counter][i]).float()
            param.data = self.master_weights[i] + noise
        self.counter += 1

    def log_reward(self, reward, cost):
        self.rewards.append(reward)
        self.costs.append(cost)
        if len(self.rewards) >= self.population_size:
            self.counter = 0
            self.evolve(np.array(self.rewards), self.c_sat)
            self.evolve(np.array(self.costs), 1 - self.c_sat)
            self.populate()
            self.rewards = []
            self.costs = []
        self.add_noise_to_weights()

    def evolve(self, functional, scale):
        # Multiply jittered weights by normalised rewards and apply to network
        if np.std(functional) != 0:
            normalized_rewards = (
                functional - np.mean(functional)) / np.std(functional)

            normalized_rewards *= scale

            # max theta
            for index, param in enumerate(self.parameters()):
                A = np.array([individual[index]
                              for individual in self.population])
                rewards_pop = torch.from_numpy(
                    np.dot(A.T, normalized_rewards).T).float()
                param.data = self.master_weights[index] + self.learning_rate / (
                    self.population_size * self.sigma) * rewards_pop
                self.master_weights[index] = param.data
