import time
import datetime
import threading
import numpy as np
import tensorflow as tf
from keras import backend as K
from a3c_model import build_policy_and_value_networks
from easy21_env import Easy21Env, V_SHAPE
from state import State
from utils import plot_V, epsilon_greedy_policy, sample_policy_action

NUM_ACTIONS = 2
STATE_SHAPE = 2

# Experiment params
NUM_CONCURRENT = 4
NUM_EPISODES = 10000

# Path params
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
EXPERIMENT_NAME = "easy21_a3c_{}_{}_{}".format(
    NUM_EPISODES, NUM_CONCURRENT, date)
SUMMARY_SAVE_PATH = "summaries/" + EXPERIMENT_NAME
SUMMARY_INTERVAL = 100

# Discounting Factor
GAMMA = 0.99

# Optimization Params
LEARNING_RATE = 0.01
DECAY = 0.99

# Global Counter
global_counter = 0
global_episode = 0


def actor_learner_thread(thread_id, env, session, graph_ops, summary_ops,
                         saver):
    # Global Counters
    global global_counter, global_episode, NUM_EPISODES

    # Unpack graph ops
    s, a, R, minimize, p_network, v_network = graph_ops

    # Unpack tensorboard summary stuff
    (r_100_summary_placeholder, update_reward_100,
     v_100_summary_placeholder, update_v_100,
     wins_100_summary_placeholder, update_wins_100,
     losses_100_summary_placeholder, update_losses_100,
     draws_100_summary_placeholder, update_draws_100,
     win_lose_ratio_100_summary_placeholder, update_win_lose_ratio_100,
     wins_total_summary_placeholder, update_wins_total,
     losses_total_summary_placeholder, update_losses_total,
     draws_total_summary_placeholder, update_draws_total,
     win_lose_ratio_total_summary_placeholder,
     update_win_lose_ratio_total, summary_op) = summary_ops

    # Set up statistics variables
    episode = 0
    r_100 = 0
    v_100 = 0
    wins_100 = 0
    losses_100 = 0
    draws_100 = 0
    wins_total = 0
    losses_total = 0
    draws_total = 0

    while episode < NUM_EPISODES:
        # Reset batch update values
        s_batch = []
        past_rewards = []
        a_batch = []

        # Set up per-episode counters
        v_ep = 0
        v_count = 0

        # Get initial state
        s_t = env.reset()
        s_t = s_t.to_arr()

        t = 0

        while s_t is not None:
            # Pick action a_t according to policy pi(a_t | s_t)
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            action_index = sample_policy_action(NUM_ACTIONS, probs)
            a_t = np.zeros([NUM_ACTIONS])
            a_t[action_index] = 1

            # Appending state and action values for batch gradient update
            s_batch.append(s_t)
            a_batch.append(a_t)

            v_ep += session.run(v_network, feed_dict={s: [s_t]})[0][0]
            v_count += 1

            # Perform action a_t to get S_t1 and r_t
            s_t1, r_t = env.step(State(s_t[0], s_t[1]), action_index)

            past_rewards.append(r_t)

            if s_t1 is not None:
                s_t1 = s_t1.to_arr()
            else:
                episode += 1
                global_counter += 1

                if global_counter == NUM_CONCURRENT:
                    global_episode += 1
                    global_counter = 0

                v_avg = v_ep / v_count
                v_100 += v_avg
                r_100 += r_t

                # Update statistics variables
                if r_t == 1:
                    wins_100 += 1
                    wins_total += 1
                elif r_t == -1:
                    losses_100 += 1
                    losses_total += 1
                else:
                    draws_100 += 1
                    draws_total += 1

                # 100 episode summary
                if episode % 100 == 0:
                    win_lose_ratio_100 = wins_100 / losses_100
                    win_lose_ratio_total = wins_total / losses_total

                    print("--------------------------------------------------")
                    print("Thread ", thread_id, "Episode ", episode)
                    print("Summary:")
                    print("Average Reward (last 100 episodes): ", r_100 / 100)
                    print("Average V (last 100 episodes): ", v_100 / 100)
                    print("Wins (last 100 episodes): ", wins_100)
                    print("Losses (last 100 episodes): ", losses_100)
                    print("Draws (last 100 episodes): ", draws_100)
                    print("Win/Lose Ratio (last 100 episodes): ",
                          win_lose_ratio_100)
                    print("Wins (total): ", wins_total)
                    print("Losses (total): ", losses_total)
                    print("Draws (total): ", draws_total)
                    print("Win/Lose Ratio (total): ", win_lose_ratio_total)
                    print("--------------------------------------------------")

                    session.run(update_reward_100, feed_dict={
                        r_100_summary_placeholder: r_100 / 100})

                    session.run(update_v_100, feed_dict={
                        v_100_summary_placeholder: v_100 / 100})

                    session.run(update_wins_100, feed_dict={
                        wins_100_summary_placeholder: wins_100})

                    session.run(update_losses_100, feed_dict={
                        losses_100_summary_placeholder: losses_100})

                    session.run(update_draws_100, feed_dict={
                        draws_100_summary_placeholder: draws_100})

                    session.run(update_win_lose_ratio_100, feed_dict={
                        win_lose_ratio_100_summary_placeholder:
                        win_lose_ratio_100})

                    session.run(update_wins_total, feed_dict={
                        wins_total_summary_placeholder: wins_total})

                    session.run(update_losses_total, feed_dict={
                        losses_total_summary_placeholder: losses_total})

                    session.run(update_draws_total, feed_dict={
                        draws_total_summary_placeholder: draws_total})

                    session.run(update_win_lose_ratio_total, feed_dict={
                        win_lose_ratio_total_summary_placeholder:
                        win_lose_ratio_total})

                    r_100 = 0
                    v_100 = 0
                    wins_100 = 0
                    losses_100 = 0
                    draws_100 = 0

            s_t = s_t1
            t += 1

        R_t = 0
        R_batch = np.zeros(t)
        for i in reversed(range(0, t)):
            R_t = past_rewards[i] + GAMMA * R_t
            R_batch[i] = R_t

        session.run(minimize, feed_dict={R: R_batch,
                                         a: a_batch,
                                         s: s_batch})


def build_graph():
    # Create shared global policy and value networks
    (s, p_network, v_network, p_params,
     v_params) = build_policy_and_value_networks(
        num_actions=NUM_ACTIONS, input_shape=STATE_SHAPE)

    # Shared global optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                          decay=DECAY)

    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    a_t = tf.placeholder("float", [None, NUM_ACTIONS])
    log_prob = tf.log(tf.reduce_sum(p_network * a_t, axis=1))
    p_loss = -log_prob * (R_t - v_network)
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + (0.5 * v_loss)

    minimize = optimizer.minimize(total_loss)
    return s, a_t, R_t, minimize, p_network, v_network


# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    reward_100 = tf.Variable(0.)
    tf.summary.scalar("average_reward_100", reward_100)
    r_100_summary_placeholder = tf.placeholder("float")
    update_reward_100 = reward_100.assign(r_100_summary_placeholder)

    v_100 = tf.Variable(0.)
    tf.summary.scalar("average_v_100", v_100)
    v_100_summary_placeholder = tf.placeholder("float")
    update_v_100 = v_100.assign(v_100_summary_placeholder)

    wins_100 = tf.Variable(0.)
    tf.summary.scalar("wins_100", wins_100)
    wins_100_summary_placeholder = tf.placeholder("float")
    update_wins_100 = wins_100.assign(wins_100_summary_placeholder)

    losses_100 = tf.Variable(0.)
    tf.summary.scalar("losses_100", losses_100)
    losses_100_summary_placeholder = tf.placeholder("float")
    update_losses_100 = losses_100.assign(losses_100_summary_placeholder)

    draws_100 = tf.Variable(0.)
    tf.summary.scalar("draws_100)", draws_100)
    draws_100_summary_placeholder = tf.placeholder("float")
    update_draws_100 = draws_100.assign(draws_100_summary_placeholder)

    win_lose_ratio_100 = tf.Variable(0.)
    tf.summary.scalar("win_lose_ratio_100)", win_lose_ratio_100)
    win_lose_ratio_100_summary_placeholder = tf.placeholder("float")
    update_win_lose_ratio_100 = win_lose_ratio_100.assign(
        win_lose_ratio_100_summary_placeholder)

    wins_total = tf.Variable(0.)
    tf.summary.scalar("wins_total", wins_total)
    wins_total_summary_placeholder = tf.placeholder("float")
    update_wins_total = wins_total.assign(wins_total_summary_placeholder)

    losses_total = tf.Variable(0.)
    tf.summary.scalar("losses_total", losses_total)
    losses_total_summary_placeholder = tf.placeholder("float")
    update_losses_total = losses_total.assign(losses_total_summary_placeholder)

    draws_total = tf.Variable(0.)
    tf.summary.scalar("draws_total", draws_total)
    draws_total_summary_placeholder = tf.placeholder("float")
    update_draws_total = draws_total.assign(draws_total_summary_placeholder)

    win_lose_ratio_total = tf.Variable(0.)
    tf.summary.scalar("win_lose_ratio_total", win_lose_ratio_total)
    win_lose_ratio_total_summary_placeholder = tf.placeholder("float")
    update_win_lose_ratio_total = win_lose_ratio_total.assign(
        win_lose_ratio_total_summary_placeholder)

    summary_op = tf.summary.merge_all()
    return (r_100_summary_placeholder, update_reward_100,
            v_100_summary_placeholder, update_v_100,
            wins_100_summary_placeholder, update_wins_100,
            losses_100_summary_placeholder, update_losses_100,
            draws_100_summary_placeholder, update_draws_100,
            win_lose_ratio_100_summary_placeholder, update_win_lose_ratio_100,
            wins_total_summary_placeholder, update_wins_total,
            losses_total_summary_placeholder, update_losses_total,
            draws_total_summary_placeholder, update_draws_total,
            win_lose_ratio_total_summary_placeholder,
            update_win_lose_ratio_total, summary_op)


def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [Easy21Env(display=False) for i in range(NUM_CONCURRENT)]

    s, a, R, minimize, p_network, v_network = graph_ops

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_SAVE_PATH, session.graph)

    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(
        target=actor_learner_thread,
        args=(
            thread_id, envs[thread_id], session, graph_ops, summary_ops,
            saver)) for thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    while global_episode < NUM_EPISODES:
        if global_episode % SUMMARY_INTERVAL == 0:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, global_episode)
    for t in actor_learner_threads:
        t.join()

    summary_str = session.run(summary_op)
    writer.add_summary(summary_str, global_episode)

    V = np.zeros(V_SHAPE)

    for i in range(1, 11):
        for j in range(1, 22):
            V[i-1, j-1] = session.run(v_network,
                                      feed_dict={s: [[i, j]]})[0][0]
    plot_V(V, save="results/" + EXPERIMENT_NAME + "_V_plot.png")


g = tf.Graph()
with g.as_default(), tf.Session() as session:
    K.set_session(session)
    graph_ops = build_graph()
    saver = tf.train.Saver()

    train(session, graph_ops, saver)
