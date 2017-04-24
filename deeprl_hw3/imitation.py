"""Functions for imitation learning."""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from keras.models import model_from_yaml
import numpy as np
import time

def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, 'r') as f:
        model = model_from_yaml(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model


def generate_expert_training_data(expert, env, num_episodes=100, render=True):
    """Generate training dataset.

    Parameters
    ----------
    expert: keras.models.Model
      Model with expert weights.
    env: gym.core.Env
      The gym environment associated with this expert.
    num_episodes: int, optional
      How many expert episodes should be run.
    render: bool, optional
      If present, render the environment, and put a slight pause after
      each action.

    Returns
    -------
    expert_dataset: ndarray(states), ndarray(actions)
      Returns two lists. The first contains all of the states. The
      second contains a one-hot encoding of all of the actions chosen
      by the expert for those states.
    """

    # Initialize states and actions.
    states = []
    actions = []

    for episode in range(num_episodes):
        done = False
        state = env.reset()

        # Gather data until episode completion
        while not done:
            if render:
                env.render()
                time.sleep(0.01)

            # Expert prediction
            action = np.argmax(expert.predict_on_batch(state[np.newaxis, ...])[0])

            # One-hot encoding of expert prediction
            train_action = np.zeros((2,))
            train_action[action] = 1.

            actions.append(train_action)
            states.append(state)

            state, reward, done, _ = env.step(action)

    return np.array(states), np.array(actions)


def test_cloned_policy(env, cloned_policy, num_episodes=50, render=True):
    """Run cloned policy and collect statistics on performance.

    Will print the rewards for each episode and the mean/std of all
    the episode rewards.

    Parameters
    ----------
    env: gym.core.Env
      The CartPole-v0 instance.
    cloned_policy: keras.models.Model
      The model to run on the environment.
    num_episodes: int, optional
      Number of test episodes to average over.
    render: bool, optional
      If true, render the test episodes. This will add a small delay
      after each action.
    """
    total_rewards = []

    for i in range(num_episodes):
        print('Starting episode {}'.format(i))
        total_reward = 0
        state = env.reset()
        if render:
            env.render()
            time.sleep(.01)
        is_done = False
        while not is_done:
            action = np.argmax(
                cloned_policy.predict_on_batch(state[np.newaxis, ...])[0])
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(.1)
        print(
            'Total reward: {}'.format(total_reward))
        total_rewards.append(total_reward)

    mean = np.mean(total_rewards)
    std = np.std(total_rewards)
    print('Average total reward: {} (std: {})'.format(
        mean, std))

    return total_rewards


def wrap_cartpole(env):
    """Start CartPole-v0 in a hard to recover state.

    The basic CartPole-v0 starts in easy to recover states. This means
    that the cloned model actually can execute perfectly. To see that
    the expert policy is actually better than the cloned policy, this
    function returns a modified CartPole-v0 environment. The
    environment will start closer to a failure state.

    You should see that the expert policy performs better on average
    (and with less variance) than the cloned model.

    Parameters
    ----------
    env: gym.core.Env
      The environment to modify.

    Returns
    -------
    gym.core.Env
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_reset = unwrapped_env._reset

    def harder_reset():
        unwrapped_env.orig_reset()
        unwrapped_env.state[0] = np.random.choice([-1.5, 1.5])
        unwrapped_env.state[1] = np.random.choice([-2., 2.])
        unwrapped_env.state[2] = np.random.choice([-.17, .17])
        return unwrapped_env.state.copy()

    unwrapped_env._reset = harder_reset

    return env

def dagger(expert, policy, env, eval_env, iterations=20, k=1):
    """Run DAGGER algorithm.

    DAGGER algorithm as described in https://katefvision.github.io/katefSlides/immitation_learning_I_katef.pdf
    on slide 26 originally shown in https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf.

    Parameters
    ----------
    expert: keras.models.Model
      Expert policy to learn policy from.
    policy: keras.models.Model
      Policy to train.
    env: gym.core.Env
      The gym environment associated with expert.
    iterations: int
      Number of iterations in which policy runs and requests expert feedback.
    k: int
      Frequency of episodes at which to evaluate policy on.

    Return
    ------
    mean_rewards, min_rewards, max_rewards
    """

    # Initialize data
    states, actions = generate_expert_training_data(expert, env, num_episodes=1, render=False)

    # Performance metrics
    mean_rewards = []
    min_rewards = []
    max_rewards = []

    for iteration in range(iterations):

        policy.fit(states, actions, epochs=50, verbose=2)
        policy_states, _ = generate_expert_training_data(policy, env, num_episodes=1, render=False)

        predictions = expert.predict_on_batch(policy_states)
        expert_argmax = np.argmax(predictions, axis=1)
        expert_actions = np.zeros(predictions.shape)
        for i in range(len(expert_actions)):
            expert_actions[i][expert_argmax[i]] = 1.
        states = np.append(states, policy_states, axis=0)
        actions = np.append(actions, expert_actions, axis=0)

        if iteration % k == 0:
            rewards = test_cloned_policy(eval_env, policy, num_episodes=100, render=False)
            mean_rewards.append(np.mean(rewards))
            min_rewards.append(np.min(rewards))
            max_rewards.append(np.max(rewards))

    return mean_rewards, min_rewards, max_rewards


