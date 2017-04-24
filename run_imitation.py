import gym
import tensorflow as tf

import deeprl_hw3.imitation as imit

POLICY_OUTPUT = "NUMBER OF EPISODES: %d\n - Final Loss: %.4f\n - Final Accuracy: %.4f\n - Base Reward: %.4f +/- %.4f\n - Hard Reward: %.4f +/- %.4f\n"
EXPERT_OUTPUT = "EXPERT:\n - Hard Reward: %.4f +/- %.4f\n"

def main():
    log_imitation()
    #test_dagger()

def test_dagger():
    with tf.Session() as sess:
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
        env = gym.make('CartPole-v0')
        policy = imit.load_model('CartPole-v0_config.yaml')
        policy.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        imit.dagger(expert, policy, env)

def log_imitation(filename='imitation_output.txt'):
    """Logs outputs for test_policy.

    Runs test_policy for 1, 10, 50, and 100 episodes.

    Outputs formatted data in file.

    Parameters
    ----------
    filename: str
      File name to be outputted to (i.e. 'imitation_output.txt')

    """
    episode_counts = [1]#[1, 10, 50, 100]
    file_string = ""
    for i in range(len(episode_counts)):
        num_episodes = episode_counts[i]
        output = test_policy(num_episodes)
        output = (num_episodes,) + output
        formatted_output = POLICY_OUTPUT % output
        file_string += formatted_output
    file_string += EXPERT_OUTPUT % evaluate_expert()
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
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
        env = gym.make('CartPole-v0')
        states, actions = imit.generate_expert_training_data(expert, env,
            num_episodes=num_episodes, render=False)
        policy = imit.load_model('CartPole-v0_config.yaml')
        policy.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = policy.fit(states, actions, epochs=50, verbose=2)

        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['acc'][-1]
        mean, std = imit.test_cloned_policy(env, policy, render=False)

        env = imit.wrap_cartpole(env)
        hard_mean, hard_std = imit.test_cloned_policy(env, policy, render=False)

        return final_loss, final_accuracy, mean, std, hard_mean, hard_std

def evaluate_expert():
    with tf.Session() as sess:
        env = gym.make('CartPole-v0')
        env = imit.wrap_cartpole(env)
        expert = imit.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
        return imit.test_cloned_policy(env, expert, render=False)

if __name__ == '__main__':
    main()