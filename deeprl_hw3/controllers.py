"""LQR, iLQR and MPC."""
import time
import pdb

import numpy as np

from scipy.linalg import solve_continuous_are


def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = x.copy()
    x_next, r, is_done, _ = env._step(u, dt)

    return (x_next - x) / dt


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))

    # approximate each column at each time step, using simulate env object
    for i in range(x.shape[0]):
        d = np.zeros((x.shape[0],))
        d[i] = delta

        x_dot_1 = simulate_dynamics(env, x + d, u, dt)
        x_dot_2 = simulate_dynamics(env, x - d, u, dt)

        A[:, i] = (x_dot_1 - x_dot_2) / 2 / delta

    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))

    # approximate each column at each time step, using simulate env object
    for i in range(u.shape[0]):
        # B[i] = (B[i] * (u + delta) - B[i] * (u - delta)) / 2 / delta
        d = np.zeros((u.shape[0],))
        d[i] = delta

        x_dot_1 = simulate_dynamics(env, x, u + d, dt)
        x_dot_2 = simulate_dynamics(env, x, u - d, dt)

        B[:, i] = (x_dot_1 - x_dot_2) / 2 / delta

    return B


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    u = np.zeros((2,))
    # u = np.random.rand(2,)

    A = approximate_A(sim_env, sim_env.state, u)  # n x n
    B = approximate_B(sim_env, sim_env.state, u)  # n x p

    P = solve_continuous_are(A, B, env.Q, env.R)
    K = np.dot(np.linalg.pinv(env.R), np.dot(B.T, P))  # n x p

    u = -np.dot(K, env.state - env.goal)  # p x 1

    return u
