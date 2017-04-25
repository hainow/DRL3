"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg

import controllers as ctl


def simulate_dynamics_next(env, x, u, step=1e-3):
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
    step: step size 
    
    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    x_next, r, is_done, _ = env._step(u, step)

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

    # adding regularization term
    # l += 10**4 * np.sum(np.square(x - env.goal))

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
    l = 10 ** 4 * np.sum(np.square(x - env.goal))

    # derivatives w.r.t x
    l_x = 2 * 10 ** 4 * (x - env.goal)  # n x 1
    l_xx = 2 * 10 ** 4 * np.eye(x.shape[0])  # n x n

    return l, l_x, l_xx


def simulate(env, x0, U, step=1e-3):
    ''' Similuate dynamics using starting state x0 and a sequence U '''
    tN = U.shape[0]

    total_cost = 0
    X = np.zeros((tN, x0.shape[0]))
    X[0] = x0

    # accumulate the cost except for the last state's
    for i in range(tN - 1):
        l, _, _, _, _, _ = cost_inter(env, X[i], U[i])
        total_cost += env.dt * l
        X[i + 1] = simulate_dynamics_next(env, X[i], U[i], step)

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
    max_iter: max iterations for optimization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """

    # access current state, x0 will be changed in MPC at different timesteps
    x0 = env.state

    U, U_new = np.zeros((tN, 2)), np.zeros((tN, 2))
    n, d = x0.shape[0], U[0].shape[0]  # n = 4, d = 2
    costs, rewards = [], []  # stats for plotting
    current_cost = np.inf
    step = 1e-3

    # begin optimizing
    for i in range(max_iter):
        sim_env.reset()
        X, new_cost = simulate(sim_env, x0, U, step)

        print("At iteration {}:\tcurrent cost = {}\tnew_cost = {}\tx0 = {}".format(i, current_cost, new_cost, x0))

        # ====================FORWARD========================
        f_u_s, f_x_s, l_u_s, l_uu_s, l_ux_s, l_x_s, l_xx_s, l_s = ilqr_forward(U, X, d, n, sim_env, tN)

        # ====================BACKWARD========================
        k, K = ilqr_backward(d, f_u_s, f_x_s, l_u_s, l_uu_s, l_ux_s, l_x_s, l_xx_s, n, tN)

        # Using k and K to get a new control sequence
        # then generate new cost
        # so that we can compare to decide whether we should stop or not
        current_x = x0.copy()
        for t in range(tN - 1):
            U_new[t] = U[t] + k[t] + np.dot(K[t], current_x - X[t])
            current_x = simulate_dynamics_next(sim_env, current_x, U_new[t])

        # collect monitoring data
        # env.render()  # optionally can render during training
        costs.append(new_cost)

        # Stopping Condition 1:
        # set stopping condition based on the cost
        if abs(new_cost - current_cost) / float(current_cost) <= float(1e-3):
            print("early stopping at loop {}, cost = {}".format(i, new_cost))
            break

        # update control sequences
        U = np.copy(U_new)
        current_cost = new_cost

        # Stopping condition 2
        # test with real env whether it can reach goal close enough in the real env.
        # if so, early stop!
        x_next = np.zeros((4,))
        env.reset()
        reward = 0.
        for u in U:
            x_next, r, _, _ = env.step(u)
            reward += r

        # collect stats
        rewards.append(reward)

        if np.sum(np.square(env.goal - x_next)) <= 1e-3:
            print("***Close enough to goal, stopping now!***")
            break

    # We change this API so that plotting will be in our driver "iLQR.py"
    # just besides U, we output plotting statistics
    return U, X, costs, rewards


# ==================================================
# ========== HELPERS for calc_ilqr_input ===========
# ==================================================
def ilqr_forward(U, X, d, n, sim_env, tN):
    """
    Forward pass of iLQR algorithm 
    
    Basically we are to create a list of cost, f(x) and their derivations for all time steps 
    
    Parameters
    ----------
    U: control sequences
    X: state sequences stemmed from control sequences 
    d: degree of freedom (similar in env and sim_env) 
    env: main environment 
    n: number of dimension in states 
    sim_env: simulate env which is used to approximate A and B to calculate f(x_t+1), df/dx, df/du 
    tN: number of time steps 
    
    Returns
    -------
    f_u_s: list of df/du (for all time steps)  
    f_x_s: list of df/dx 
    l_u_s: list of costs  
    l_uu_s: list of d^2l/du^2 
    l_ux_s: list of d(dl/du)/dx 
    l_x_s: list of dl/dx
    l_xx_s: list of d^2l/dx^2 
    """
    # ===== calculate all derivations ====
    f_x_s = np.zeros((tN, n, n))
    f_u_s = np.zeros((tN, n, d))
    l_s = np.zeros((tN, 1))
    l_x_s = np.zeros((tN, n))
    l_xx_s = np.zeros((tN, n, n))
    l_u_s = np.zeros((tN, d))
    l_uu_s = np.zeros((tN, d, d))
    l_ux_s = np.zeros((tN, d, n))

    # populate cost and derivations
    for t in range(tN - 1):
        # we need A and B for derivation of f(x_t+1) = A*x_t + B*u_t
        A = ctl.approximate_A(sim_env, X[t], U[t])
        B = ctl.approximate_B(sim_env, X[t], U[t])
        f_x_s[t] = A * sim_env.dt + np.eye(n)  # magic trick
        f_u_s[t] = B * sim_env.dt

        # then for normal costs and their derivations
        l, l_x, l_xx, l_u, l_uu, l_ux = cost_inter(sim_env, X[t], U[t])
        l_s[t] = l * sim_env.dt
        l_x_s[t] = l_x * sim_env.dt
        l_xx_s[t] = l_xx * sim_env.dt
        l_u_s[t] = l_u * sim_env.dt
        l_uu_s[t] = l_uu * sim_env.dt
        l_ux_s[t] = l_ux * sim_env.dt

    # and separately for final state
    l_s[-1], l_x_s[-1], l_xx_s[-1] = cost_final(sim_env, X[tN - 1])

    return f_u_s, f_x_s, l_u_s, l_uu_s, l_ux_s, l_x_s, l_xx_s, l_s


def ilqr_backward(d, f_u_s, f_x_s, l_u_s, l_uu_s, l_ux_s, l_x_s, l_xx_s, n, tN):
    '''
    Backward part of iLQR algorithm 
    
    Calcualte Q which makes use of derivations of V
    From them we can calculate k and K for updating U
    
    Paramters
    ---------
    Refer to ilqr_forward. Basically it consumes the output of the forward part  

    Returns
    -------
    k amd K: gain matrices which will be used to update control sequences U 
    '''

    V_x, V_xx = l_x_s[-1].copy(), l_xx_s[-1].copy()  # get values for the first backward step (tN -1)
    k, K = np.zeros((tN, d)), np.zeros((tN, d, n))
    for t in range(tN - 2, -1, -1):
        Q_x = l_x_s[t] + np.dot(f_x_s[t].T, V_x)
        Q_u = l_u_s[t] + np.dot(f_u_s[t].T, V_x)
        Q_xx = l_xx_s[t] + np.dot(f_x_s[t].T, np.dot(V_xx, f_x_s[t]))
        Q_ux = l_ux_s[t] + np.dot(f_u_s[t].T, np.dot(V_xx, f_x_s[t]))
        Q_uu = l_uu_s[t] + np.dot(f_u_s[t].T, np.dot(V_xx, f_u_s[t]))

        # inverse
        Q_uu_inv = np.linalg.pinv(Q_uu + np.eye(Q_uu.shape[0]))  # use Devin's trick

        # update k and K
        k[t] = -np.dot(Q_uu_inv, Q_u)
        K[t] = -np.dot(Q_uu_inv, Q_ux)

        # update V, V_x, V_xx for next step in backward
        V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
        V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

    return k, K
