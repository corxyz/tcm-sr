import math
import numpy as np
import pylab as plt
from numpy.matlib import repmat
from numpy import cumsum, nanmean, nansum
import util, pachinko_sim, bias_var, pstopSim

########################################
#  CONSTANTS & HYPERPARAMETERS
########################################

cm = 1/2.54  # centimeters in inches

n_row, n_col = 10, 9
n_rew = 15
min_rew, max_rew = 5, 24
rew_size = 1
add_absorb_state = True
extra_trans_on_sr = True

n_exp = 100
n_trial = 50
n_samp = 1000
n_rep = 50
max_samp = 100
n_traj = 10000
rho, beta = 1, 0
alpha = 0.01
lamb = 0.7
gamma = 0.9
pstop = 0.05
rew_mod_alpha = 0.5
lr_schedule_type = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.001}}

########################################
#  HELPER FN
########################################

def run_sim(n_trial, n_samp, n_row, n_col, n_rew, maze, s, T, M, gamma, pstop=None,
            MFC=None, max_samp=50, rho=None, beta=None, rew_size=1, 
            add_absorb_state=True, extra_trans_on_sr=True, verbose=False):
  '''
    TCM-SR sample-based evaluation (general, can use for all beta values)

    Params:
      n_trial: number of trials (rollouts) per experiment (int)
      n_samp: number of samples drawn per experiment (int)
      n_row: number of rows (int)
      n_col: number of columns (int)
      n_rew: number of rewards (int)
      maze: Plinko board (numpy array)
      s: starting position (same across experiments) (int tuple)
      T: one-step transition matrix (numpy array)
      M: successor representation of the specified gamma (numpy array)
      gamma: discount factor (optional, float)
      pstop: interruption probability (optional, None or float)
      MFC: item-to-context association matrix, 
            specifying which context vector is used to update the current context vector once an item is recalled
            (optional, numpy array)
      max_samp: max number of samples to draw (optional, int)
      rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
      rew_size: reward magnitude (optional, number)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
      extra_trans_on_sr: whether to count visit upon entering or exiting a state (optional, bool)
                              (if using the definition in Zhou et al. (2024), this should be set to True)
      verbose: if true, prints progress (% complete) (optional, bool)
    
    Return:
      vtrue: true values of s0 in each experiment (numpy array)
      vsamp: sample-based estimated value of s0 in each experiment (numpy array)
      samped: indices of sampled states (numpy array)
  '''
  if beta == 0:
    rew_idx = np.ravel_multi_index(np.where(maze == rew_size), (n_row, n_col))  # reward locations
    vtrue, _, vsamp, _, samped = bias_var.run_sim(n_trial, n_samp, n_row, n_col, rew_idx.size, 
                                                  s, M, M,rew_size=rew_size, userandom=False, idx=rew_idx, MFC=MFC,
                                                  add_absorb_state=add_absorb_state, rho=rho, beta=beta, verbose=verbose)
  else:
    eGamma = pstop * gamma + (1-pstop)   # effective gamma
    M_effect = util.get_sr(T, gamma=eGamma, extra_trans_on_sr=extra_trans_on_sr)
    samped, vsamp, _, vtrue = pstopSim.run_sim(n_rep, n_trial, n_row, n_col, n_rew, s, M, M_effect, pstop, userandom=False, maze=maze, 
                                          MFC=MFC, max_samp=max_samp, rho=rho, beta=beta)
  return vtrue, vsamp, samped

def get_rollout_samped_val(vsamp, samped, beta):
  '''
    Compute the average sample-based action value estimates from rollouts

    Params:
      vtrue: true values of s0 in each experiment (numpy array)
      samped: indices of sampled states (numpy array)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (float)
    
    Return:
      mean: average sample-based action value estimates as a function of time (samples) (numpy array)
  '''
  n_rep, n_trial, n_samp = vsamp.shape
  mean = np.zeros((n_rep, n_trial * n_samp))
  for e in range(n_rep):
    s = 0
    while s < n_trial * n_samp:
      t = math.ceil((s+1) / n_samp)
      r = (s+1) % n_samp
      if samped[e,t-1,r-1] <= 0:   # the current trial ended, start inspecting the next
        if t < n_trial:
          t += 1
          r = 1
        else:
          break
      if r == 0: 
        r = n_samp
      if t == 1: 
        est = np.expand_dims(nansum(vsamp[e,t-1,:r])*beta, axis=0)
      else: 
        est = np.append(nansum(vsamp[e,:t-1,:], axis=1)*beta, np.expand_dims(nansum(vsamp[e,t-1,:r])*beta, axis=0), axis=0)
      mean[e,s] = nanmean(est)
      s += 1
    mean[e,s:] = mean[e,s-1]
  return mean

def get_samp_based_reward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=0):
  '''
    Compute the expected total reward from the sample-based choices, assuming a greedy policy

    Param:
      vtrue_s1: true values of action 1 in each experiment & gamma (numpy array)
      vtrue_s2: true values of action 2 in each experiment & gamma (numpy array)
      vsamp_s1: sampled-based value estimates of action 1 in each experiment & gamma (numpy array)
      vsamp_s2: sampled-based value estimates of action 2 in each experiment & gamma (numpy array)
      samped_s1: indices of sampled states by performing action 1 (numpy array)
      samped_s2: indices of sampled states by performing action 2 (numpy array)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
    
    Return:
      values: expected true action values (numpy array)
      choice: model choices based on its action value estimates (numpy array)
      exp_rew: expected total reward from the sampled-based choices (numpy array)
  '''
  if beta == 0:   # i.i.d. samples
    n_trial, n_samp = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, n_samp, 1).T
    vtrue_s2 = repmat(vtrue_s2, n_samp, 1).T
    v1 = cumsum(vsamp_s1, axis=1)/np.arange(1, n_samp+1)
    v2 = cumsum(vsamp_s2, axis=1)/np.arange(1, n_samp+1)
  else:    # generalized rollout
    _, n_trial, n_samp = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, n_trial * n_samp, 1).T
    vtrue_s2 = repmat(vtrue_s2, n_trial * n_samp, 1).T
    v1 = get_rollout_samped_val(vsamp_s1, samped_s1, beta)
    v2 = get_rollout_samped_val(vsamp_s2, samped_s2, beta)
  values = np.transpose(np.array([vtrue_s1, vtrue_s2]), (1,2,0))    # expected option values
  choice = np.where(v1 >= v2, 0, 1)                                # sample-based choice
  exp_rew = np.zeros(choice.shape)
  for i in range(choice.shape[0]):
    for j in range(choice.shape[1]):
      exp_rew[i,j] = values[i,j,choice[i,j]]
  return values, choice, exp_rew

def get_perf(values, exp_rew):
  '''
    Compute the fraction of maximum available rewards obtained by the model, assuming a greedy policy

    Params:
      values: expected true action values (numpy array)
      exp_rew: expected total reward from the sampled-based choices (numpy array)
  '''
  max_val = np.max(values, axis=-1)
  assert(max_val.shape == exp_rew.shape)
  return np.nanmean(exp_rew / max_val, axis=(0,1))

def plot_samp_based_perf(avg, rews):
  '''
    Plot the fraction of maximum available rewards obtained by the model, assuming a greedy policy

    Params:
      avg: average performance (numpy array)
      rews: number of rewards (numpy array)
  '''
  nr, n_samp = avg.shape
  fig, ax = plt.subplots(1, figsize=[5*cm, 2.8*cm], dpi=300)
  for i in range(nr):
    ax.semilogx(np.arange(0,n_samp), avg[i], linewidth=1, linestyle='--', label=str(rews[i]) + ' rewards')

  ax.tick_params(labelsize=5, length=2, pad=0)
  ax.set_xlabel('Number of samples', fontdict={'fontsize':5}, labelpad=2)
  ax.set_ylabel('% of max rewards', fontdict={'fontsize':5}, labelpad=2)
  ax.set_ylim(0,1)
  ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)

  fig.subplots_adjust(right=0.6)
  plt.tight_layout(pad=0.25)
  plt.savefig('mod-dash.pdf', dpi=300)
  plt.show()

def update_sr(k, alpha, alphaMod, scheduler, maze, traj, M_init, 
              gamma, lamb, add_absorb_state, extra_trans_on_sr):
  '''
    Use TD-learning to update the successor representation given a new observation (trajectory)

    Params:
      k: trajectory number
      alpha: initial learning rate (optional, float)
      alphaMod: reward modulated learning rate (float)
      scheduler: learning rate schedule (dictionary)
      maze: Plinko board (numpy array)
      traj: a trajectory (list)
      M_init: initial SR (numpy array)
      gamma: discount factor (float)
      lamb: eligibility trace lambda (float)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
      extra_trans_on_sr: whether to count visit upon entering or exiting a state (optional, bool)
                      (if using the definition in Zhou et al. (2024), this should be set to True)
    
    Return:
      MTD: TD-updated successor representation (numpy array)
      alpha: learning rate adjusted according to the schedule (float)
      alphaMod: reward modulated learning rate according to the schedule (float)
  '''
  if lr_schedule_type == 'step':
    if (k+1) % scheduler['epochs'] == 0: # apply step decay
      alpha *= scheduler['rate']
      if alphaMod: 
        alphaMod *= scheduler['rate']
  elif lr_schedule_type == 'exp':
    alpha *= np.exp(-scheduler['rate'] * k)
    if alphaMod: 
      alphaMod *= np.exp(-scheduler['rate'] * k)
  MTD = bias_var.get_sr_td(maze, traj, M_init, alpha=alpha, gamma=gamma, lamb=lamb, 
                          rew_thres=0, rew_mod_alpha=alphaMod,
                          add_absorb_state=add_absorb_state, extra_trans_on_sr=extra_trans_on_sr)
  return MTD, alpha, alphaMod

def samp_based_val_est(maze, n_samp, s, M, MTD, alphaMod):
  '''
    Compute the sample-based action value estimates using TCM-SR

    Params:
      maze: Plinko board (numpy array)
      n_samp: number of samples drawn per experiment (int)
      gamma: discount factor (optional, float)
      s: starting position (same across experiments) (int tuple)
      M: successor representation of the specified gamma (numpy array)
      MTD: TD-updated successor representation (numpy array)
      alphaMod: reward modulated learning rate (float)
    
    Return:
      vtrue: true values of s0 in each experiment (numpy array)
      vsamp: sample-based estimated value of s0 in each experiment (numpy array)
      samped: indices of sampled states (numpy array)
  '''
  n_row, n_col = maze.shape
  MTF_id = np.identity(M.shape[0])
  _, vtrue, vsamp, _, samped = bias_var.run_sim_iw(n_rep, n_samp, n_row, n_col, maze,
                                                    s, MTD, M, MFC=MTF_id, add_absorb_state=add_absorb_state, 
                                                    rho=rho, beta=beta, verbose=False, mod=alphaMod, mod_frac=0)
  return vtrue, vsamp, samped

def learn_sr(maze, s, scheduler):
  '''
    Learn the successor representation from a set of trajectories

    Params:
      maze: Plinko board (numpy array)
      s: starting position (same across experiments) (int tuple)
      scheduler: learning rate schedule (dictionary)
    
    Return:
      MTD: learned successor representation (numpy array)
  '''
  n_state = maze.size + add_absorb_state
  for t in range(n_traj):
    # generate trajectories
    traj = pachinko_sim.get_rand_traj(maze, s, add_absorb_state=add_absorb_state)
    # obtain intermediate SR
    if t == 0: MTD = np.identity(n_state)
    MTD, lr, alphaMod = update_sr(t, alpha, rew_mod_alpha, scheduler, maze, traj, 
                                  MTD, gamma, lamb, add_absorb_state, extra_trans_on_sr)
  return MTD

def samp_intermediate_sr(maze, s, M, scheduler):
  '''
    Get sample-based action estimate from an intermediate SR

    Params:
      maze: Plinko board (numpy array)
      s: starting position (same across experiments) (int tuple)
      M: successor representation of the specified gamma (numpy array)
      scheduler: learning rate schedule (dictionary)
    
    Return:
      vtrue: true values of s0 in each experiment (numpy array)
      vsamp: sample-based estimated value of s0 in each experiment (numpy array)
      samped: indices of sampled states (numpy array)
  '''
  if n_traj < 5: 
    vtrue, vsamp, samped = np.zeros(n_trial), np.zeros((n_trial, n_samp)), np.zeros((n_trial, n_samp))
    for j in range(n_trial):
      MTD = learn_sr(maze, s, scheduler)
      vtrue[j], vsamp[j,:], samped[j,:] = samp_based_val_est(maze, n_samp, s, M, MTD, rew_mod_alpha)
  else: 
    MTD = learn_sr(maze, s, scheduler)
    vtrue, vsamp, samped = samp_based_val_est(maze, n_samp, s, M, MTD, rew_mod_alpha)
  return vtrue, vsamp, samped

########################################
#  MAIN ROUTINES
########################################

def main_converged_sr():
  '''
    Simulate TCM-SR choice performace using fully learned (converged) SRs
  '''
  if beta == 0: 
    correct, choice = np.zeros((n_exp, n_trial, n_samp, 2)), np.zeros((n_exp, n_trial, n_samp))
  else: 
    correct = np.zeros((n_exp, n_rep, n_trial * max_samp, 2))
    choice = np.zeros((n_exp, n_rep, n_trial * max_samp))
  # pick two states
  s1, s2 = (0,4), (0,5)
  # compute true transition matrix and SR
  n_state = n_row * n_col
  T = pachinko_sim.init_trans(n_state, n_row, n_col, add_absorb_state=add_absorb_state)
  M = util.get_sr(T, gamma=gamma, extra_trans_on_sr=extra_trans_on_sr)
  MTF_id = np.identity(M.shape[0])
  rews = [1, 5, 10, 20]
  avg = np.zeros((len(rews), choice.shape[-1]))
  for i, nr in enumerate(rews):
    for e in range(n_exp):
      if n_exp >= 10 and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
      # generate each maze
      if gamma == 0 and (beta == 0 or pstop == 1):
        maze = pachinko_sim.init_maze(n_row, n_col, nr, reachable=[s1,s2], rew_size=rew_size, userandom=True, force_row=1)
      else:
        maze = pachinko_sim.init_maze(n_row, n_col, nr, reachable=[s1,s2], rew_size=rew_size, userandom=True)
      # draw samples from s1                             
      vtrue_s1, vsamp_s1, samped_s1 = run_sim(n_trial, n_samp, n_row, n_col, n_rew, maze, s1, T, M, gamma,
                                  pstop=pstop, MFC=MTF_id, max_samp=max_samp, rho=rho, beta=beta, rew_size=rew_size)
      # draw samples from s2
      vtrue_s2, vsamp_s2, samped_s2 = run_sim(n_trial, n_samp, n_row, n_col, n_rew, maze, s2, T, M, gamma,
                                  pstop=pstop, MFC=MTF_id, max_samp=max_samp, rho=rho, beta=beta, rew_size=rew_size)
      # get correct responses and sample-based choices
      correct[e,:], _, choice[e,:] = get_samp_based_reward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=beta)
    # compute overall correctness rate (average case)
    avg[i,:] = get_perf(correct, choice)
  # plot graph
  plot_samp_based_perf(avg, rews)

def main_intermediate_sr():
  '''
    Simulate TCM-SR choice performace using incrementally learned (intermediate) SRs
  '''
  if n_traj < 5: 
    correct, choice = np.zeros((n_exp, n_trial, n_samp, 2)), np.zeros((n_exp, n_trial, n_samp))
  else: 
    correct, choice = np.zeros((n_exp, n_rep, n_samp, 2)), np.zeros((n_exp, n_rep, n_samp))
  # pick two states
  s1, s2 = (0,4), (0,5)
  # compute true transition matrix and SR
  n_state = n_row * n_col
  T = pachinko_sim.init_trans(n_state, n_row, n_col, add_absorb_state=add_absorb_state)
  M = util.get_sr(T, gamma=gamma, extra_trans_on_sr=extra_trans_on_sr)
  scheduler = schedulers[lr_schedule_type]
  rews = [1, 5, 10, 20]
  avg = np.zeros((len(rews), choice.shape[-1]))
  for i, nr in enumerate(rews):
    for e in range(n_exp):
      if n_exp >= 10 and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
      # generate each maze
      if gamma == 0 and (beta == 0 or pstop == 1):
        maze = pachinko_sim.init_maze(n_row, n_col, nr, reachable=[s1,s2], rew_size=rew_size, userandom=True, force_row=1)
      else:
        maze = pachinko_sim.init_maze(n_row, n_col, nr, reachable=[s1,s2], rew_size=rew_size, userandom=True)
      # draw samples from s1                             
      vtrue_s1, vsamp_s1, samped_s1 = samp_intermediate_sr(maze, s1, M, scheduler)
      # draw samples from s2
      vtrue_s2, vsamp_s2, samped_s2 = samp_intermediate_sr(maze, s2, M, scheduler)
      # get correct responses and sample-based choices
      correct[e,:], _, choice[e,:] = get_samp_based_reward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=beta)
    # compute overall correctness rate (average case)
    avg[i,:] = get_perf(correct, choice)
  # plot graph
  plot_samp_based_perf(avg, rews)
