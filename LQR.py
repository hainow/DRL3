import gym
import pdb

import numpy as np
import matplotlib.pyplot as plt

import deeprl_hw3.controllers as ctl

plt.rcParams['figure.figsize'] = 15, 8

"""
Infinite-horizon, continuous-time LQR

Ref: https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
"""


def plot_states_and_control(states=None, u=None, env_name=None):
    assert (len(states) == len(u))
    x = [i + 1 for i in range(len(states))]

    figure, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=False)
    figure.suptitle(env_name, fontsize=20)

    ax1.plot(x, states[:, 0], color='b')
    ax1.set_ylabel("q[0]", color='b')

    ax2.plot(x, states[:, 1], color='b')
    ax2.set_ylabel("q[1]", color='b')

    ax3.plot(x, states[:, 2], color='r')
    ax3.set_ylabel("dq[0]", color='r')

    ax4.plot(x, states[:, 3], color='r')
    ax4.set_ylabel("dq[1]", color='r')

    ax5.plot(x, u[:, 0], color='g')
    ax5.set_ylabel("u[0]", color='g')

    ax6.plot(x, u[:, 1], color='g')
    ax6.set_ylabel("u[1]", color='g')
    ax6.set_xlabel("iterations", fontsize=14)

    plt.savefig(env_name + ".png")
    plt.show()


def control_lqr(env_name="TwoLinkArm-v0"):
    env, sim_env = gym.make(env_name), gym.make(env_name)
    # env.state, sim_env.state = env.reset(), sim_env.reset()  # no need

    rewards, states, us = [], [], []
    count = 0
    for _ in range(10000):
        count += 1
        u = ctl.calc_lqr_input(env, sim_env)
        x_next, r, is_done, _ = env._step(u)  # state is updated in _step(u) already

        # collect monitoring stats
        rewards.append(r)
        states.append(env.state)
        us.append(u)

        # show time
        env.render()

        # early stopping
        if np.sum(np.square(env.goal - env.state)) <= 1e-3:
            print("\n\nConverged! Early stopping at loop {}... \n".format(count))
            break

    print(rewards)
    print("Total rewards: {}".format(np.sum(rewards)))

    # plotting
    states, us = np.array(states), np.array(us)
    plot_states_and_control(states, us, "LQR: " + env_name)


if __name__ == "__main__":
    env_names = [
        'TwoLinkArm-v0',
        'TwoLinkArm-limited-torque-v0',
        'TwoLinkArm-v1',
        'TwoLinkArm-limited-torque-v1',
    ]

    try:
        print("\nQuestion 1")
        control_lqr(env_names[0])

        print("\nQuestion 2")
        control_lqr(env_names[1])

        print("\nQuestion 3 is to compare q1 vs. q2")

        print("\nQuestion 4")
        control_lqr(env_names[2])

        print("\nQuestion 5")
        control_lqr(env_names[3])

        print("\nQuestion 6 is to compare q4 vs. q5")

    except KeyboardInterrupt:
        exit(0)
