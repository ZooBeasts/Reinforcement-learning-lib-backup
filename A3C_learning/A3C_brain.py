import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gymnasium as gym
import torch.multiprocessing as mp
import torch.nn.functional as F

class ActorCriticNett(nn.Module):
    def __init__(self,
                 n_actions,
                 n_features) -> None:
        super(ActorCriticNett, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features

        self.pi1 = nn.Linear(self.n_features, 128)
        self.pi2 = nn.Linear(128, self.n_actions)

        self.v1 = nn.Linear(self.n_features, 128)
        self.v2 = nn.Linear(128, 1)

        self.distribution = torch.distributions.Categorical

    def weight_init(self):
        for layer in [self.pi1, self.pi2, self.v1, self.v2]:
            nn.init.normal_(layer.weight, 0.0, 0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        pi1 = F.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = torch.softmax(logits, 1).detach()
        m = self.distribution(prob)

        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = torch.softmax(logits, 1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep,
                 global_ep_r, res_queue, name,
                 n_features, n_actions):
        super().__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_squeue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = ActorCriticNett(n_features, n_actions)
        self.max_ep = 3000
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.update_global_iter = 5
        self.gamma = 0.9

    def run(self):
        total_step = 1
        while self.g_ep.value < self.max_ep:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                if self.name == "w00":
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[np.newaxis,:]))

                s_, r, done, _ = self.env.step(a)
                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global_iter == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done,
                                  s_, buffer_s, buffer_a, buffer_r, self.gamma)

                buffer_s, buffer_a, buffer_r = [], [], []

                if done:  # done and print information
                    record(self.g_ep, self.g_ep_r, ep_r,
                           self.res_queue, self.name)
                    break
                s = s_
                total_step += 1

        self.res_queue.put(None)


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    if len(np_array.shape) == 1:  # Check for 1D array
        return torch.from_numpy(np_array)
    else:
        return torch.from_numpy(np_array).squeeze(0)


def push_and_pull(opt, lnet, gnet, done, s_, bs, br, ba, gamma):
    if done:
        v_s_ = 0.
    else:
        v_s_ = lnet(v_wrap(s_[None]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(
            np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )



class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr,
                                         betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

if __name__ == "__main__":


    rl = Worker.run