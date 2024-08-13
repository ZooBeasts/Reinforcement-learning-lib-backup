import gymnasium as gym
import A3C_brain
import torch.multiprocessing as mp
import os
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    env = gym.make('CartPole-v1',render_mode='human')
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n

    gnet = A3C_brain.ActorCriticNett(N_S, N_A)
    # https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-multiprocessing/
    gnet.share_memory()

    opt = A3C_brain.SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))

    global_ep, global_ep_r, res_queue = mp.Value(
        "i", 0), mp.Value('d', 0), mp.Queue()

    # PARALLEL TRAINING

    workers = [A3C_brain.Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, N_S, N_A)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]