import csv
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import deeprl_hw3.imitation as imit
import deeprl_hw3.reinforce as reinforce

def create_callback(env, output):
    """Create callback for REINFORCE.
    
    Used to write data into .csv file for REINFORCE.

    Parameters
    ----------
    env: gym.core.Env
      Environment being run on.
    output: str
      Output .csv file.
    """
    def callback(iteration, reward, model):
        if iteration == 0:
            f = open(output, 'a+')
            f.write("Iteration,Mean,Min,Max\n")
            f.close()
        if iteration % 10 == 0:
            rewards = imit.test_cloned_policy(env, model, num_episodes=100, render=False)
            f = open(output, 'a+')
            f.write("%d,%.4f,%.4f,%.4f\n" % (iteration, np.mean(rewards), np.min(rewards), np.max(rewards)))
            f.close()
    return callback

def test_reinforce(output='reinforce_data.csv'):
    """Get metrics for REINFORCE algorithm.

    Gets necessary data to answer q1 and q2 in Question 3.

    Parameters
    ----------
    output: str
      Name of file to write evaluation data on base env to.
    """
    env = gym.make('CartPole-v0')
    cb = create_callback(env, output)
    with tf.Session() as sess:
        model = reinforce.reinforce(env, sess, callback=cb)

        env = imit.wrap_cartpole(env)
        rewards = imit.test_cloned_policy(env, model, render=False)
        print("Hard Reward: %.4f +/- %.4f" % (np.mean(rewards), np.std(rewards)))
        f = open("reinforce_output.txt", 'a+')
        f.write("REINFORCE:\n - Hard Reward: %.4f +/- %.4f\n" % (np.mean(rewards), np.std(rewards)))
        f.close()

def plot_reinforce(dataname='reinforce_data.csv', plotname='REINFORCE Plot.png'):
    """Plot the REINFORCE learning curve.

    Plot contains the mean reward with error bars representing
    minimum and maximum reward.  Needed to answer q1 of Question 3's
    extra creddit portion for REINFORCE.

    Parameters
    ----------
    dataname: str
      Name of csv file containing reward data for a DAGGER learning curve.
    plotname: str
      Name of png file to write plot to.

    """

    indices = []
    min_rewards = []
    max_rewards = []
    mean_rewards = []

    with open(dataname) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            indices.append(int(row['Iteration']))
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
    plt.title("REINFORCE Average Rewards vs. Episode with Min/Max Error Bars")
    plt.xticks(np.arange(0, 20, 2))
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode')
    plt.savefig(plotname)

def main():
    test_reinforce()
    plot_reinforce()

if __name__ == '__main__':
    main()
