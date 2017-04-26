"""LQR, iLQR and MPC."""

import numpy as np

import ilqr
import ilqr_fast


def calc_mpc_input(env, sim_env, tN=50, max_iter=1000000):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_iter: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    U_mpc = np.zeros((tN, 2))
    X_mpc = np.zeros((tN, 4))
    env.reset()
    sim_env.reset()

    for t in range(tN):
        print("\n***Begin using iLQR to optimize for current timestep {}***\n".format(t))
        U, _, _, _ = ilqr.calc_ilqr_input(env, sim_env, tN, max_iter)

        # update x_t and record u_t for next timestep's optimization
        current_x, _, _, _ = env.step(U[0])
        env.state = current_x

        # update optimal control sequence at current timestep, discard the rest
        U_mpc[t] = U[0]
        X_mpc[t] = current_x

    return U_mpc, X_mpc


def faster_calc_mpc_input(env, sim_env, tN=50, max_iter=1000000, n_groups=2):
    """Calculate the optimal control input for the given state.
    
    A FASTER SOLUTION: instead of optimizing just 1 state at a time 
     We optimize a group of continuous steps. 

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_iter: max iterations for optmization
    n_groups: number of groups. For example, in order to get 100 control steps, we divide it into 10 groups
                We run 10 times of iLQR, each time we optimize for 100 steps, 90 steps, 80 steps, ... and so on
    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    U_mpc = np.zeros((tN, 2))
    X_mpc = np.zeros((tN, 4))
    env.reset()
    sim_env.reset()

    for t in range(n_groups):
        print("\n***Begin using iLQR to optimize for current timestep {}/{} ***\n".format(t, (n_groups-1)))
        U, _, _, _ = ilqr_fast.calc_ilqr_input(env, sim_env, tN - t * (tN/n_groups), max_iter)

        # re-assign starting env.
        env.state = X_mpc[t*(tN/n_groups)]

        # update x_t and record u_t for next timestep's optimization
        for i in range(tN/n_groups):
            current_x, _, _, _ = env.step(U[i])
            env.state = current_x
            X_mpc[t * (tN/n_groups) + i] = current_x

        # update optimal control sequence at current timestep, discard the rest
        U_mpc[t * (tN/n_groups): (t+1) * (tN/n_groups)] = U[:tN/n_groups]

    return U_mpc, X_mpc
