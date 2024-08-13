import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss()

    def _build_net(self):
        self.policy = nn.Sequential(
            nn.Linear(self.n_features, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, self.n_actions),
            nn.Softmax(dim=-1)
        )


    def choose_action(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        # Convert the observation to a tensor and add batch dimension
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        prob_weights = self.policy(observation_tensor)
        action = torch.multinomial(prob_weights, num_samples=1)
        return action.item()
        # if isinstance(observation, tuple):
        #     observation = observation[0]  # Extract the array if it's a tuple
        # if observation is None or np.isscalar(observation):
        #     raise ValueError("Received invalid observation.")
        #
        # print("Observation:", observation)
        # # observation = observation[0]
        # # if observation is None or len(observation) == 0:
        # #     raise ValueError("Received invalid observation.")
        # observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        # prob_weights = self.policy(observation)
        # action = torch.multinomial(prob_weights, num_samples=1)
        # return action.item()

    def store_transition(self, s, a, r):
        print("Pre-conversion observation:", s)
        if isinstance(s, tuple):
            s = s[0]
        if not isinstance(s, np.ndarray):
            raise TypeError("Observation is not a numpy array")

        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        if len(self.ep_obs) > 0:
            print("Observation stack shapes:", [o.shape for o in self.ep_obs])  # Debugging line
            ep_obs = torch.tensor(np.stack(self.ep_obs), dtype=torch.float32)
        else:
            raise ValueError("No observations to learn from.")
        # ep_obs = torch.tensor(self.ep_obs, dtype=torch.float32)
        ep_as = torch.tensor(self.ep_as, dtype=torch.int64)
        discounted_ep_rs_norm = torch.tensor(self._discount_and_norm_rewards(), dtype=torch.float32)

        # Zero gradients before running the backpropagation
        self.optimizer.zero_grad()

        # Forward pass to get logits
        logits = self.policy(ep_obs)

        # Compute Cross Entropy Loss (without applying softmax to logits)
        loss = self.loss_func(logits, ep_as)

        # Multiply the loss by the normalized, discounted rewards
        loss = (loss * discounted_ep_rs_norm).mean() # Define your loss function appropriately
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


if __name__ == '__main__':

    n_actions = 3
    n_features = 4
    model = PolicyGradient(n_actions, n_features)
    for _ in range(5):
        observation = np.random.rand(n_features)
        action = model.choose_action(observation)
        reward = np.random.randn()
        model.store_transition(observation, action, reward)

        # Learn from the gathered data
    model.learn()
    print("Test completed successfully!")
