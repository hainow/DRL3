import csv
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import deeprl_hw3.imitation as imit

POLICY_OUTPUT = "NUMBER OF EPISODES: %d\n - Final Loss: %.4f\n - Final Accuracy: %.4f\n - Base Reward: %.4f +/- %.4f\n - Hard Reward: %.4f +/- %.4f\n"
DAGGER_OUTPUT = "DAGGER:\n - Hard Reward: %.4f +/- %.4f\n"
EXPERT_OUTPUT = "EXPERT:\n - Hard Reward: %.4f +/- %.4f\n"


def main():
    log_imitation()
    test_dagger()
    plot_dagger()


def plot_dagger(dataname='dagger_data.csv'):
    """Plot the DAGGER learning curve.

    Plot contains the mean reward with error bars representing
    minimum and maximum reward.  Needed to answer q1 of Question 2's
    extra creddit portion for DAGGER.

    Parameters
    ----------
    dataname: str
      Name of csv file containing reward data for a DAGGER learning curve.

    """

    indices = [i for i in range(20)]
    min_rewards = []
    max_rewards = []
    mean_rewards = []

    with open(dataname) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            min_rewards.append(float(row['Min']))
            max_rewards.append(float(row['Max']))
            mean_rewards.append(float(row['Mean']))

    mean_rewards = np.array(mean_rewards)
    min_rewards = np.array(min_rewards)
    max_rewards = np.array(max_rewards)

    upper = max_rewards - mean_rewards
    lower = mean_rewards - min_rewards

    plt.figure()
    plt.errorbar(indices, mean_rewards, yerr=[lower, upper], ecolor='red', elinewidth=0.4, capsize=4)
    plt.title("DAGGER Average Rewards vs. Episode with Min/Max Error Bars")
    plt.xticks(np.arange(0, 20, 2))
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode')
    plt.savefig('DAGGER Plot.png')


def test_dagger(filename='imitation_output.txt', dataname='dagger_data.csv'):
    """Get metrics for DAGGER algorithm.

    Gets necessary data to answer q1 and q2 in the extra credit portion (DAGGER)
    in Question 2.

    Parameters
    ----------
    filename: str
      Name of file to append DAGGER performance on wrapper environment to.
    dataname: str
      Name of file to write evaluation data on base env to.
    """
    with tf.Session() as sess:
        # Load expert
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')

        # Initialize environments
        env = gym.make('CartPole-v0')
        eval_env = gym.make('CartPole-v0')
        eval_env = imit.wrap_cartpole(eval_env)

        # Initialize policy model.
        policy = imit.load_model('CartPole-v0_config.yaml')
        policy.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Run DAGGER
        mean_rewards, min_rewards, max_rewards = imit.dagger(expert, policy, env, eval_env)

        # Test on wrapper environment.
        rewards = imit.test_cloned_policy(eval_env, policy, render=False)
        hard_mean = np.mean(rewards)
        hard_std = np.std(rewards)

        # append to file
        f = open(filename, 'a+')
        f.write(DAGGER_OUTPUT % (hard_mean, hard_std))
        f.close()

        # convert data to .csv format
        data_string = "Mean,Min,Max\n"
        for i in range(len(mean_rewards)):
            data_string += "%.4f,%.4f,%.4f\n" % (mean_rewards[i], min_rewards[i], max_rewards[i])

        # write data to file
        f = open(dataname, 'w')
        f.write(data_string)
        f.close()


def log_imitation(filename='imitation_output.txt'):
    """Logs outputs for test_policy.

    Runs test_policy for 1, 10, 50, and 100 episodes.

    Outputs formatted data in file.

    Parameters
    ----------
    filename: str
      File name to be outputted to (i.e. 'imitation_output.txt')

    """
    episode_counts = [1, 10, 50, 100]
    file_string = ""

    for i in range(len(episode_counts)):
        num_episodes = episode_counts[i]

        # get performance values of policy
        output = test_policy(num_episodes)

        # format output
        output = (num_episodes,) + output
        formatted_output = POLICY_OUTPUT % output
        file_string += formatted_output

    # Add expert performance for benchmark
    file_string += EXPERT_OUTPUT % evaluate_expert()

    # write to file
    f = open(filename, 'w')
    f.write(file_string)
    f.close()


def test_policy(num_episodes):
    """Train and test imitation-based policy.

    Parameters
    ----------
    num_episodes: int
      Number of episodes to generate data for imitation policy.

    Returns
    -------
    final loss, final accuracy, mean reward, reward std, wrapper mean reward, wrapper reward std
    """
    with tf.Session() as sess:
        # load expert and policy model
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
        policy = imit.load_model('CartPole-v0_config.yaml')
        policy.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # initialize environment
        env = gym.make('CartPole-v0')

        # generate data from expert
        states, actions = imit.generate_expert_training_data(expert, env,
                                                             num_episodes=num_episodes, render=False)

        # train policy
        history = policy.fit(states, actions, epochs=50, verbose=2)

        # get performance values
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['acc'][-1]
        rewards = imit.test_cloned_policy(env, policy, render=False)
        mean = np.mean(rewards)
        std = np.std(rewards)

        env = imit.wrap_cartpole(env)
        hard_rewards = imit.test_cloned_policy(env, policy, render=False)
        hard_mean = np.mean(hard_rewards)
        hard_std = np.std(hard_rewards)

        return final_loss, final_accuracy, mean, std, hard_mean, hard_std


def evaluate_expert():
    """Evaluate expert on the wrapper environment.
    Return
    -----
    mean(rewards), std(rewards)
    """
    with tf.Session() as sess:
        env = gym.make('CartPole-v0')
        env = imit.wrap_cartpole(env)
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
        rewards = imit.test_cloned_policy(env, expert, render=False)
        return np.mean(rewards), np.std(rewards)


if __name__ == '__main__':
    main()
