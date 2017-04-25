import gym
import time
import matplotlib.pyplot as plt

import deeprl_hw3.ilqr as ilqr
# import deeprl_hw3.ilqr_sped_up as ilqr

plt.rcParams['figure.figsize'] = 15, 8


def plot_states_and_control_ilqr(states=None, u=None, env_name=None):
    assert (len(states) == len(u))
    x = [i + 1 for i in range(len(states))]

    figure, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=False)
    # figure.suptitle(env_name, fontsize=20)

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
    ax6.set_xlabel("timesteps", fontsize=14)

    plt.savefig(env_name + ".png")
    plt.show()


def plot_costs_ilqr(costs=None, env_name=None):
    x = [i+1 for i in range(len(costs))]
    plt.plot(x, costs, color='r', marker='*', label='')
    # plt.title("iLQR: Training costs over timesteps for " + env_name, fontsize=20)
    plt.xlabel("iterations", fontsize=14)
    plt.ylabel("cost values")
    plt.grid()
    plt.legend()
    plt.savefig("cost_" + env_name)
    plt.show()


def plot_rewards_ilqr(rewards=None, env_name=None):
    x = [i + 1 for i in range(len(rewards))]
    plt.plot(x, rewards, color='g', marker='*', label='')
    # plt.title("iLQR: Training rewards over timesteps for " + env_name, fontsize=20)
    plt.xlabel("iterations", fontsize=14)
    plt.ylabel("reward values")
    plt.grid()
    plt.legend()
    plt.savefig("reward_" + env_name)
    plt.show()


def show_optimal_trajectory(env, U):
    env.reset()
    reward = 0.
    for u in U:
        _, r, _, _ = env.step(u)
        reward += r
        env.render()
        time.sleep(0.1)
    return reward


def control_ilqr(env_name="TwoLinkArm-v0"):

    env, sim_env = gym.make(env_name), gym.make(env_name)
    U, X, costs, rewards = ilqr.calc_ilqr_input(env, sim_env, tN=100, max_iter=1000000)
    plot_costs_ilqr(costs, "iLQR: " + env_name)
    plot_rewards_ilqr(rewards, "iLQR: " + env_name)
    plot_states_and_control_ilqr(X, U, "iLQR: " + env_name)

    print("\nShowing optimal trajectory")
    final_reward = show_optimal_trajectory(env, U)
    print("Total Reward for optimal trajectory: {}".format(final_reward))


if __name__ == "__main__":
    env_names = [
        'TwoLinkArm-v0',
        'TwoLinkArm-v1',
    ]

    try:
        print("\nQuestion 1")
        control_ilqr(env_names[0])

        print("\nQuestion 2")
        control_ilqr(env_names[1])

        print("\nQuestion 3 is to compare q1 vs. q2")

    except KeyboardInterrupt:
        exit(0)
