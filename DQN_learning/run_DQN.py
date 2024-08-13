from DQN_learning.Maze import Maze
from DQN_brain import DeepQNetwork


def update():
    step = 0

    for episode in range(300):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if step > 200 & step % 5 ==0:
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions,
                      env.n_features,
                      learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.001,batch_size=32,
                      double_q=True)
    env.after(100, update)
    env.mainloop()
    # RL.plot_cost()