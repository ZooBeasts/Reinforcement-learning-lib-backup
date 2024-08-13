import gymnasium as gym
from PG_learning.PG_brain import PolicyGradient
from PG_learning.agent import PolicyGradient, PG_agent
import matplotlib.pyplot as plt
import time


render = True
display_reward_threshold = 1

env = gym.make('CartPole-v1',render_mode='human')
# env = env.unwrapped
# obs, info= env.reset(seed=1)
# env.reset()

#
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)


# RL = PolicyGradient(
#     n_actions = env.action_space.n,
#     n_features = env.observation_space.shape[0],
#     learning_rate = 0.2,
#     reward_decay = 0.9)

RL = PG_agent(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.2,
    reward_decay = 0.9)

for i_episode in range(3000):
    observation = env.reset()

    while True:
        if render:
            env.render()
        action = RL.choose_action(observation)
        result = env.step(action)
        # print('result form env.step():', result)
        observation_, reward, done, info = result[:4]
        # print('Result from env.step():', observation_, reward, done, info)
        if isinstance(observation_, tuple):

            observation_ = observation_[0]
        RL.store_transition(observation, action, reward)

        if done:
            episode_rewards_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = episode_rewards_sum
            else:
                running_reward = running_reward * 0.99 + episode_rewards_sum * 0.01

            if running_reward > display_reward_threshold:
                render = True
            print("episode:", i_episode, " reward:", int(running_reward))

            vt = RL.learn()

            # if i_episode == 0:
            #     plt.plot(vt)
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_

env.close()
