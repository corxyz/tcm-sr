import math
import numpy as np
from numpy.random import rand, randn
from numpy import percentile, nanmean, nansum
from numpy.matlib import repmat
import pachinko_sim

########################################
#  CONSTANT
########################################

cm = 1/2.54  # centimeters in inches

########################################
#  HELPER FN
########################################

def run_sim(n_exp, n_trial, n_row, n_col, n_rew, s0, M, M_effect, pstop, 
            MFC=None, max_samp=50, add_absorb_state=True, userandom=True, maze=None, 
            rand_rew=False, reachable=None, rew_bias=None,
            rho=None, beta=None, check_context_unit_norm=True, verbose=False):
  '''
  TCM-SR sample-based evaluation (most useful for beta > 0, i.e., non-i.i.d. sampling)

  Params:
    n_exp: number of experiments/boards (int)
    n_trial: number of trials (rollouts) per experiment (int)
    n_row: number of rows (int)
    n_col: number of columns (int)
    n_rew: number of rewards (int)
    s0: starting position (same across experiments) (int tuple)
    M: successor representation of the specified gamma (numpy array)
    M_effect: successor representation of the effective gamma (numpy array)
    pstop: interruption probability (optional, None or float)
    MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
          (optional, numpy array)
    max_samp: max number of samples to draw (optional, int)
    add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    userandom: use random reward placement (optional, bool)
    maze: Plinko board (optional, numpy array)
    rand_rew: use random extra reward on top of the specified reward magnitude (optional, bool)
    reachable: only place rewards in states reachable from the specified state(s) 
               (optional, None, int tuple, or list of int tuples)
    rew_bias: probabilities of an arbitrary reward being in the left vs right side of the board (optional, list)
    rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
    beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
    check_context_unit_norm: if true, checks the norm of the updated context vector is 1 (optional, bool)
    verbose: if true, prints progress (% complete) (optional, bool)
  
  Return:
    samped_row: sampled rows (numpy array)
    vtrue: true value of s0 w.r.t. the true gamma in each experiment (numpy array)
    vsamp: sample-based estimated value of s0 in each experiment (numpy array)
    veffect: true value of s0 w.r.t. the effective gamma in each experiment (numpy array)
  '''
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(n_exp)
  veffect = np.zeros(n_exp)
  vsamp = np.zeros((n_exp, n_trial, max_samp))
  samped_row = np.negative(np.ones((n_exp, n_trial, max_samp), dtype=int))

  # compute rho and beta (if one of them was not provided aka None)
  if rho == None:
    rho = 1-beta
  elif beta == None:
    beta = 1-rho

  for e in range(n_exp):
    if verbose and n_exp >= 10 and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
    if userandom and rew_bias is None:
      maze = pachinko_sim.init_maze(n_row, n_col, n_rew, userandom=True, reachable=reachable)
    elif rew_bias is not None:
      maze = pachinko_sim.init_biased_maze(n_row, n_col, n_rew, bias=rew_bias)
    rvec = maze.flatten() # 1-step rewards
    if rand_rew: rvec = 1 + randn(rvec.size)
    if add_absorb_state: rvec = np.append(rvec, 0)

    state_idx = np.arange(0, maze.size).reshape(maze.shape)    # state indices
    stim = np.identity(maze.size + add_absorb_state)

    vtrue[e] = np.dot(M[state_idx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    veffect[e] = np.dot(M_effect[state_idx[s0],:], rvec)        # Compute the effective value function (as v=Mr)

    for t in range(n_trial):
      c = stim[:,state_idx[s0]] # starting context vector (set to the starting state)
      samp_idx = np.negative(np.ones(max_samp, dtype=int))
      # roll out with a probability of stopping (pstop)
      stopped = False
      i = 0
      while not stopped:
        # define sampling distribution
        a = np.matmul(M.T, c)
        P = a / np.sum(a)
        assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

        # draw one sample
        tmp = np.where(rand() <= np.cumsum(P))[0]
        samp_idx[i] = tmp[0]
        if samp_idx[i] >= n_row * n_col: # entered absorbing state
          stopped = True
          break
        else:
          samped_row[e,t,i], _ = np.unravel_index(samp_idx[i], (n_row, n_col))
          f = stim[:,samp_idx[i]]

        cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
        cIN = cIN1/np.linalg.norm(cIN1)
        assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'
        
        # update context
        c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
        c = c/np.linalg.norm(c)
        if check_context_unit_norm:
          assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
        
        # do I stop now?
        stopped = stopped | (rand() <= pstop)
        i += 1
      
      # compute total sampled rewards
      rewsamp = rvec[samp_idx[samp_idx >= 0]]
      rewsamp = np.append(rewsamp, np.zeros(max_samp - len(rewsamp)))
      vsamp[e,t,:] = rewsamp * np.sum(M[state_idx[s0],:]) # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)    

  return samped_row, vsamp, vtrue, veffect

def get_val_est_pctl(vsamp, vtrue, beta, percentiles=[2.5, 25, 50, 75, 97.5], avg_trials=False):
  '''
    Compute the mean and percentile of the sample-based action value estimates 
    as a function of the first n samples

    Params:
      vsamp: sample-based action value estimate in each experiment (numpy array)
      vtrue: corresponding true values (numpy array)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (float)
      percentiles: list of percentile points (optional, list)
      avg_trials: whether to average across trials or not (optional, bool)
    
    Return:
      mean: average action value estimate as a function of samples (numpy array)
      pctl: percentiles of the action value estimate as a function of samples (numpy array)
      bias: bias (avg estimate - truth) as a function of samples (numpy array)
  '''
  n_exp, n_trial, n_samp = vsamp.shape
  if beta == 0:
    bias = vsamp - repmat(vtrue, n_samp, 1).T
    mean = np.empty(n_samp)
    pctl = np.empty((len(percentiles), n_samp))
    for s in range(n_samp):
      mean[s] = nanmean(nanmean(bias[:,:s+1],axis=1))
      pctl[:,s] = percentile(nanmean(bias[:,:s+1],axis=1), percentiles)
  elif avg_trials: # average across trials
    mean = np.empty(n_trial)
    pctl = np.empty((len(percentiles), n_trial))
    bias = np.empty((n_exp, n_trial))
    for t in range(n_trial):
      bias[:,t] = nanmean(nansum(vsamp[:,:t+1,:], axis=2), axis=1) - vtrue
      mean[t] = nanmean(bias[:,t])
      pctl[:,t] = percentile(bias[:,t], percentiles)
  else:
    mean = np.empty(n_trial*n_samp)
    pctl = np.empty((len(percentiles), n_trial*n_samp))
    bias = np.empty((n_exp, n_trial*n_samp))
    for s in range(n_trial*n_samp):
      t = math.ceil((s+1) / n_samp) # compute number of trials included
      r = (s+1) % n_samp            # compute number of samples included that belong to the last (possibly incomplete) trial
      if r == 0: r = n_samp
      if t == 1: est = np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1)
      else: est = np.append(nansum(vsamp[:,:t-1,:], axis=2)*beta, np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1), axis=1)
      bias[:,s] = nanmean(est, axis=1) - vtrue
      mean[s] = nanmean(bias[:,s])
      pctl[:,s] = percentile(bias[:,s], percentiles)
  return mean, pctl, bias
