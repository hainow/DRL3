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

def reinforce(env, sess):
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf.initialize_all_variables()
    opt = tf.train.AdamOptimizer()

    #action_input = tf.placeholder(dtype=tf.int32, shape=(1,))
    #loss = tf.log(tf.gather(tf.reshape(model.output, (2,)), [action_input]))
    loss = tf.log(model.output)
    grads = opt.compute_gradients(loss, model.weights)

    #print(grads)

    def get_gradient(state, grad):
        return sess.run(grad, feed_dict = {
                model.input : state
            })

    alpha = 0.025
    gamma = 0.9

    rewards = []
    while True:
        S, A, R = run_episode(env, model)

        reward = sum(R)
        rewards.append(reward)
        print("REWARD: %.4f" % reward)
        if len(rewards) > 20 and np.std(np.array(rewards[-5:])) < 5.:
            print("CONVERGED")
            break

        G = process_rewards(R, gamma)
        total_gradient = None
        for t in range(len(S)):
            gradient = [None] * len(grads)
            for i in range(len(grads)):
                gradient[i] = (-1 * (gamma ** t) * alpha * G[t] * get_gradient(S[t].reshape((1,4)), grads[i][0]), grads[i][1])

            opt.apply_gradients(gradient)

    return get_total_reward(env, model)
