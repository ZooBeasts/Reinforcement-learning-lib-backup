import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyGradient(nn.Module):
    def __init__(self,
                 n_actions: float,
                 n_features: float,
                 learning_rate=0.01):
        super(PolicyGradient, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.policy = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.Tanh(),
            nn.Linear(10, self.n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.policy(x)


class PG_agent:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.PolicyGradient = PolicyGradient(n_actions, n_features, learning_rate)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.optimizer = optim.Adam(self.PolicyGradient.policy.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss()

    def choose_action(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        prob_weights = self.PolicyGradient(observation)
        action = torch.multinomial(prob_weights, num_samples=1)
        return action.item()

    def store_transition(self, s, a, r):
        if isinstance(s, tuple):
            s = s[0]
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        if len(self.ep_obs) > 0:
            # print("Observation stack shapes:", [o.shape for o in self.ep_obs])  # Debugging line
            ep_obs = torch.tensor(np.stack(self.ep_obs), dtype=torch.float32)
        else:
            raise ValueError("No observations to learn from.")
        # ep_obs = torch.tensor(self.ep_obs, dtype=torch.float32)
        ep_as = torch.tensor(self.ep_as, dtype=torch.int64)
        discounted_ep_rs_norm = torch.tensor(self._discount_and_norm_rewards(), dtype=torch.float32)
        # Zero gradients before running the backpropagation
        self.optimizer.zero_grad()
        # Forward pass to get logits
        logits = self.PolicyGradient.policy(ep_obs)
        # Compute Cross Entropy Loss (without applying softmax to logits)
        loss = self.loss_func(logits, ep_as)
        # Multiply the loss by the normalized, discounted rewards
        loss = (loss * discounted_ep_rs_norm).mean()  # Define your loss function appropriately
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_model(self, filename="pg_model.pt"):
        model_state = {
            'policy_state_dict': self.PolicyGradient.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(model_state, filename)

    def load_model(self, filename="pg_model.pt"):
        """Loads a previously saved model state."""
        model_state = torch.load(filename)
        self.PolicyGradient.policy.load_state_dict(model_state['policy_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])


if __name__ == '__main__':
    n_actions = 3
    n_features = 4
    agent = PG_agent(n_actions, n_features)

    for _ in range(5):
        observation = np.random.rand(n_features)
        action = agent.choose_action(observation)
        reward = np.random.randn()
        agent.store_transition(observation, action, reward)

    agent.learn()
    print("Test completed successfully!")
