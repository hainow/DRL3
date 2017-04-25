import gym
import numpy as np
import tensorflow as tf

import deeprl_hw3.imitation as imit


def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    _, _, rewards = run_episode(env, model)
    return sum(rewards)


def run_episode(env, model, render=False):
    states = []
    actions= []
    rewards = []
    #print('Starting episode')
    state = env.reset()
    if render:
        env.render()
        time.sleep(0.01)
    is_done = False
    while not is_done:
        states.append(state)
        probability, action = choose_action(model, state)
        actions.append(action)
        state, reward, is_done, _ = env.step(action)
        if render:
            env.render()
            time.sleep(0.1)
        #print('Total reward: {}'.format(total_reward))
        rewards.append(reward)
    return states, actions, rewards

def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """

    # model prediction
    action_prob = model.predict_on_batch(observation[np.newaxis, ...])[0][0]
    action = 0 if np.random.uniform() < action_prob else 1
    return action_prob, action

def process_rewards(rewards, gamma):
    discounted = [0.]*len(rewards)
    discounted[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        discounted[i] = rewards[i] + gamma*discounted[i+1]
    return np.array(discounted)

def reinforce(env, sess, gamma = 0.98, alpha = 0.00025):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment
    sess: tf.Session

    Returns
    -------
    total_reward: float
    """

    model = imit.load_model('CartPole-v0_config.yaml')
    tf.initialize_all_variables()

    action_input = tf.placeholder(dtype=tf.int32, shape=(1,))
    loss = tf.log(tf.gather(tf.reshape(model.output, (2,)), [action_input]))

    grads = tf.gradients(loss, model.weights)

    def get_gradient(state, action):
        return sess.run(grads, feed_dict = {
                model.input : state,
                action_input : [action]
            })

    rewards = []
    iteration = 0
    while True:
        S, A, R = run_episode(env, model)

        reward = sum(R)
        rewards.append(reward)
        print("REWARD (%d): %.4f" % (iteration, reward))
        if len(rewards) > 20 and np.std(np.array(rewards[-20:])) < 3. and np.mean(np.array(rewards[-5:])) > 50.:
            print("CONVERGED")
            return get_total_reward(env, model)

        G = process_rewards(R, gamma)
        total_gradient = None
        weights = model.get_weights()
        for t in range(len(S)):
            gradients = get_gradient(S[t].reshape((1,4)), A[t])
            assert(len(weights) == len(gradients))
            for i in range(len(weights)):
                #weights[i] += (gamma ** t) * alpha * G[t] * gradients[i]
                weights[i] += alpha * G[t] *(gamma**t)* gradients[i]
        model.set_weights(weights)
        iteration += 1
