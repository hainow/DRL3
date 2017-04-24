"""LQR, iLQR and MPC."""

import numpy as np

import controllers as ctl

import ilqr

def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    x_next, r, is_done, _ = env.step(u)

    return x_next


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    # main intermediate cost
    l = np.sum(np.square(u))

    # derivatives w.r.t x are all Zeros
    l_x = np.zeros(x.shape[0])
    l_xx = np.zeros((x.shape[0], x.shape[0]))

    # derivatives w.r.t u
    l_u = 2 * u
    l_uu = 2 * np.eye(u.shape[0])

    # derivative w.r.t. u and x
    l_ux = np.zeros((u.shape[0], x.shape[0]))

    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    l = 10**4 * np.sum(np.square(x - env.goal))

    # derivatives w.r.t x
    l_x = 2 * 10**4 * (x - env.goal)  # n x 1
    l_xx = 2 * 10**4 * np.eye(x.shape[0])  # n x n

    return l, l_x, l_xx


def simulate(env, x0, U):
    ''' Similuate dynamics using starting state x0 and a sequence U '''
    tN = U.shape[0]

    total_cost = 0
    X = np.zeros((tN, x0.shape[0]))
    X[0] = x0

    # accumulate the cost except for the last state's
    for i in range(tN - 1):
        l, _, _, _, _, _ = cost_inter(env, U[i], X[i])
        total_cost += env.dt * l
        X[i + 1] = simulate_dynamics_next(env, X[i], U[i])

    # append last state's cost
    ll, _, _ = cost_final(env, X[tN - 1])
    total_cost += ll

    return X, total_cost


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1000000):
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
    for t in range(tN):
        print("\n***Begin using iLQR to optimize for current timestep {}***\n".format(t))
        U, _, _ = ilqr.calc_ilqr_input(env, sim_env, tN, max_iter)

        # reset for next iLQR iterative process
        env.reset()
        sim_env.reset()

        # update x_t and record u_t for next timestep's optimization
        current_x, _, _, _ = env.step(U[0])
        env.state = current_x

        # update optimal control sequence at current timestep, discard the rest
        U_mpc[t] = U[0]
        X_mpc[t] = current_x

    return U_mpc, X_mpc

