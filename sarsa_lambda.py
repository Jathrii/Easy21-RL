import numpy as np
from easy21_env import Easy21Env, V_SHAPE, Q_SHAPE
from utils import get_epsilon, epsilon_greedy_policy, plot_V, save_nd_arr,
load_nd_arr


class SarsaLambda:

    def __init__(self, num_episodes, gamma, lmbd):
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lmbd = lmbd

        self.reset()

    def reset(self):
        self.Q = np.zeros(Q_SHAPE)
        self.env = Easy21Env()
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def learn(self):
        N0 = 100
        Ns = np.zeros(V_SHAPE)
        Nsa = np.zeros(Q_SHAPE)

        for i in range(self.num_episodes):

            print("Episode:" + str(i + 1))

            E = np.zeros(Q_SHAPE)

            self.env.reset()

            print("Your card: " + str(self.env.state.player_sum))
            print("Dealer's card: " + str(self.env.state.dealer_card))

            state1 = self.env.state

            epsilon = get_epsilon(N0, Ns, state1)

            action1 = epsilon_greedy_policy(epsilon, self.Q, state1)

            while (state1 is not None):
                index1 = (state1.dealer_card-1, state1.player_sum-1, action1)
                Q1 = self.Q[index1]

                state2, reward = self.env.step(state1, action1)

                if (state2 is not None):
                    action2 = epsilon_greedy_policy(epsilon, self.Q, state2)
                    index2 = (state2.dealer_card-1,
                              state2.player_sum-1, action2)
                    Q2 = self.Q[index2]
                else:
                    Q2 = 0
                    if reward == 1:
                        self.wins += 1
                    elif reward == -1:
                        self.losses += 1
                    else:
                        self.draws += 1

                delta = reward + self.gamma * Q2 - Q1

                E[index1] += 1
                Ns[index1[0:2]] += 1
                Nsa[index1] += 1

                alpha = 1 / Nsa[index1]

                self.Q += alpha * delta * E
                E *= self.gamma * self.lmbd

                state1 = state2
                if (state2 is not None):
                    action1 = action2
                    epsilon = get_epsilon(N0, Ns, state1)

            print("-------------------------------------------------------")

        info = "Wins: {}\nLosses: {}\nWin to Lose ratio: {}\nDraws: {}"
        info = info.format(self.wins, self.losses, (self.wins / self.losses),
                           self.draws)
        print(info)

        path = "results/sarsa_{}_{}".format(self.lmbd, self.num_episodes)

        with open(path + '_info.txt', 'a') as the_file:
            the_file.write(info)

        save_nd_arr(path + "_Q.txt", self.Q)

        plot_V(np.max(self.Q, axis=2), save=path + "_V_plot.png")

# Training 200000 episodes for each parameter value
# lambda in {0, 0.1, 0.2, ..., 1}
lmbd_values = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}

for lmbd in lmbd_values:
    agent = SarsaLambda(200000, 1, lmbd)
    agent.learn()