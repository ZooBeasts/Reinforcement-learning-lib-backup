import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, n_features,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500, batch_size=32,
                 e_greedy_increment=None,
                 double_q=True ):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.double_q = double_q
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2+2))
        self.eval_net = self._build_network()
        self.target_net = self._build_network()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0

    def _build_network(self):
        model = nn.Sequential(
            nn.Linear(self.n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_actions)
        )
        return model

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # Ensure all elements have the same dimensionality

        # Memory Reshaping
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            self.memory = np.array([transition]) # Initialize memory
        else:
            self.memory = np.vstack((self.memory, transition))

        self.memory_counter += 1

        # transition = torch.from_numpy(np.hstack((s, [a, r], s_))).float()
        # index = self.memory_counter % self.memory_size
        # self.memory[index] = transition
        # self.memory_counter += 1
        # print(transition, transition.shape, np.array(s).shape, a, r, np.array(s_).shape,r==-1)



    def choose_action(self, observation):
        observation = np.array(observation)  # Ensure it's a numpy array
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Debugging: Print the shape of the observation tensor
        # print("Observation Tensor Shape:", observation_tensor.shape)

        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(observation_tensor)
            action = torch.argmax(actions_value).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action
        # observation = np.array(observation)
        # observation = torch.tensor(observation[np.newaxis, :])
        # if np.random.uniform() < self.epsilon:
        #     with torch.no_grad():
        #         actions_value = self.eval_net(observation)
        #     action = torch.argmax(actions_value).item()
        # else:
        #     action = np.random.randint(0, self.n_actions)
        # return action

    def learn(self):
        if self.memory_counter < self.batch_size:
            return  # Not enough memory to learn, exit the function

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

            # Choose indices from available samples
        if self.memory_counter > self.memory_size:
            sample_index = torch.randperm(self.memory_size)[:self.batch_size]
        else:
            sample_index = torch.randperm(self.memory_counter)[:self.batch_size]
        batch_memory = self.memory[sample_index, :]

        # Compute Q values
        q_next, q_eval4next = self.get_q_next_and_q_eval(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32))

        # Set up Q-learning targets
        q_target = q_eval.clone()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        actions = batch_memory[:, self.n_features].astype(int)  # Actions taken
        rewards = batch_memory[:, self.n_features + 1]  # Immediate rewards

        if self.double_q:
            # Double Q-learning logic
            max_act4next = torch.argmax(q_eval4next, dim=1)  # Selecting actions from eval net
            selected_q_next = q_next[batch_index, max_act4next]  # Getting corresponding Q values from target net
        else:
            selected_q_next = torch.max(q_next, dim=1)[0]  # Standard Q-learning

        # Reshape for correct broadcasting
        selected_q_next = selected_q_next.view(self.batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(self.batch_size, 1)

        q_target[
            batch_index, actions] = (rewards + self.gamma * selected_q_next).squeeze()  # Update Q values only for taken actions

        # Loss computation and backpropagation
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self._update_epsilon()
    def get_q_next_and_q_eval(self, states):
        states_tensor = torch.tensor(states, dtype=torch.float32)  # Convert to tensor
        with torch.no_grad():
            return self.target_net(states_tensor), self.eval_net(states_tensor)

    def _update_epsilon(self):
        if self.epsilon_increment is not None and self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment





if __name__ == '__main__':

    RL = DeepQNetwork(n_actions=4,
                      n_features=4,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.001,
                      batch_size=32).learn()
