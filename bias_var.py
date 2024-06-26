import numpy as np
from numpy.random import rand, randn, random
from numpy import nanmean, nansum, percentile
import pylab as plt
import pachinko_sim

########################################
#  CONSTANT
########################################

cm = 1/2.54  # centimeters in inches

########################################
#  HELPER FN
########################################

def init_maze(n_row, n_col, n_rew):
  '''
    Initialize a rectangular Plinko board

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      n_rew: number of rewards (int)
  '''
  return pachinko_sim.init_maze(n_row, n_col, n_rew, userandom=True)

def get_sr_td(maze, traj, M_init, alpha=0.1, gamma=1, lamb=0, 
            add_absorb_state=True, extra_trans_on_sr=True, rew_thres=None, rew_mod_alpha=None):
  '''
    Compute SR based on observations using TD-learning, with reward modulated learning if specified

    Params:
      maze: Plinko board (numpy array)
      traj: a trajectory (list)
      M_init: initial SR (numpy array)
      alpha: initial learning rate (optional, float)
      gamma: discount factor (optional, float)
      lamb: eligibility trace lambda (optional, float)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
      extra_trans_on_sr: whether to count visit upon entering or exiting a state (optional, bool)
                      (if using the definition in Zhou et al. (2024), this should be set to True)
      rew_thres: reward threshold - only modulate learning rate if passed (optional, None or number)
      rew_mod_alpha: reward modulated learning rate (optional, None or number)
  
    Return:
      M: successor representation of the specified gamma (numpy array)
  '''
  n_row, n_col = maze.shape
  n_state = maze.size
  M = M_init.copy()                  # SR
  e = np.zeros(n_state+add_absorb_state)             # eligibility trace
  for n, curPos in enumerate(traj[:-add_absorb_state]):
    cur_idx = np.ravel_multi_index(curPos,(n_row,n_col))
    next_pos = traj[n+extra_trans_on_sr]
    if next_pos[0] == n_row:  # absorbing state
      next_idx = maze.size
    else:
      next_idx = np.ravel_multi_index(next_pos,(n_row,n_col))
    # modulate learning rate if necessary
    if rew_mod_alpha and next_idx < maze.size and maze[next_pos] > rew_thres:
      lr = rew_mod_alpha
    else:
      lr = alpha
    s, s1 = np.zeros(n_state+add_absorb_state), np.zeros(n_state+add_absorb_state)
    s[cur_idx] = 1
    s1[next_idx] = 1
    sT, s1T = np.expand_dims(s, axis=1).T, np.expand_dims(s1, axis=1).T
    e = gamma * lamb * e + s
    M = M + lr * np.matmul(np.expand_dims(e, axis=1), (s1T + gamma * np.matmul(s1T, M) - np.matmul(sT, M)))
  return M

def get_sr_heb(maze, traj, M_init, alpha=0.1, gamma=1, lamb=0, add_absorb_state=True,
                  extra_trans_on_sr=True, rew_thres=None, rew_mod_alpha=None):
  '''
    Compute SR based on observations using Hebbian rule, with reward modulated learning if specified

    Params:
      maze: Plinko board (numpy array)
      traj: a trajectory (list)
      M_init: initial SR (numpy array)
      alpha: initial learning rate (optional, float)
      gamma: discount factor (optional, float)
      lamb: eligibility trace lambda (optional, float)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
      extra_trans_on_sr: whether to count visit upon entering or exiting a state (optional, bool)
                      (if using the definition in Zhou et al. (2024), this should be set to True)
      rew_thres: reward threshold - only modulate learning rate if passed (optional, None or number)
      rew_mod_alpha: reward modulated learning rate (optional, None or number)
  
    Return:
      M: successor representation of the specified gamma (numpy array)
  '''
  n_row, n_col = maze.shape
  n_state = maze.size
  M = M_init.copy()                  # SR
  e = np.zeros(n_state+add_absorb_state)             # eligibility trace
  for n, curPos in enumerate(traj[:-add_absorb_state]):
    cur_idx = np.ravel_multi_index(curPos,(n_row,n_col))
    next_pos = traj[n+extra_trans_on_sr]
    if next_pos[0] == n_row:  # absorbing state
      next_idx = maze.size
    else:
      next_idx = np.ravel_multi_index(next_pos,(n_row,n_col))
    # modulate learning rate if necessary
    if rew_mod_alpha and next_idx < maze.size and maze[next_pos] > rew_thres:
      lr = rew_mod_alpha
    else:
      lr = alpha
    s, s1 = np.zeros(n_state+add_absorb_state), np.zeros(n_state+add_absorb_state)
    s[cur_idx] = 1
    s1[next_idx] = 1
    sT, s1T = np.expand_dims(s, axis=1).T, np.expand_dims(s1, axis=1).T
    e = gamma * lamb * e + s
    M = M + lr * np.matmul(np.expand_dims(e, axis=1), s1T) - lr * np.matmul(np.matmul(np.expand_dims(s, axis=1), sT), M)
  return M

def draw_samp(s0, maze, state_idx, stim, M, n_samp, 
              rho=None, beta=None, add_absorb_state=True, MFC=np.identity(0),
              check_context_unit_norm=True, plot_samp=True, incl_last_samp=False):
  '''
    Draw and plot i.i.d. samples using TCM-SR

    Params:
      s0: starting position (same across experiments) (int tuple)
      maze: Plinko board (numpy array)
      state_idx: state index (numpy array)
      stim: starting context vector of each state (numpy array)
      M: successor representation of the specified gamma (numpy array)
      n_samp: number of samples drawn per experiment (int)
      rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
      MFC: item-to-context association matrix, 
           specifying which context vector is used to update the current context vector once an item is recalled
           (optional, numpy array)
      check_context_unit_norm: if true, ensure all context vectors are unit vectors (optional, bool)
      plot_samp: if true, plot out each sample (optional, bool)
      incl_last_samp: if true, plot the previous sample with each sample (optional, bool)
  '''
  if MFC.shape != M.shape:
    MFC = np.identity(M.shape[0]) # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  if add_absorb_state:
    M = M[:-1, :-1]
    MFC = MFC[:-1, :-1]
    stim = stim[:-1, :-1]

  c = stim[:,state_idx[s0]] # starting context vector (set to the starting state)

  samp_idx = np.negative(np.ones(n_samp, dtype=int))
  n_row, n_col = maze.shape
  n_row_plot, n_col_plot = 1, 4
  fig = plt.figure(figsize=[8.8*cm, 3*cm], dpi=300)
  for i in range(n_samp):
    # define sampling distribution
    a = np.matmul(M.T, c)
    P = a / np.sum(a)
    assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

    # draw sample
    tmp = np.where(rand() <= np.cumsum(P))[0]
    samp_idx[i] = tmp[0]
    f = stim[:,samp_idx[i]]        # sample vector
    cIN = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
    cIN = cIN/np.linalg.norm(cIN)

    # update context
    c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
    c = c/np.linalg.norm(c)
    if check_context_unit_norm:
      assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # plot
    if plot_samp and i < n_row_plot * n_col_plot:
      axes = plt.subplot(n_row_plot, n_col_plot, i+1)
      # draw the board
      title = r'$\mathbf{x}$'
      axes.set_title(title + r'$_{}$'.format(str(i+1)), size=7, pad=2)
      axes.set_xticks(np.arange(-.5, n_col, 1))
      axes.set_yticks(np.arange(-.5, n_row, 1))
      axes.tick_params(length=0, labelbottom=False, labelleft=False)   
      axes.grid()
      axes.set_aspect('equal', adjustable='box')
      # plot sampling distribution
      im = plt.imshow(np.reshape(P, maze.shape), cmap="Greys")

      # plot sample
      sRow,sCol = np.unravel_index(samp_idx[i], maze.shape)
      axes.plot(sCol, sRow, 'c*', markersize=5, label="sample")

      if incl_last_samp:
        # mark previous sample
        if i==0:
          prevRow, prevCol = s0[0], s0[1]
        else:
          prevRow, prevCol = np.unravel_index(samp_idx[i-1], maze.shape)
        axes.plot(prevCol, prevRow, 'c*', markersize=5, markerfacecolor='none', markeredgewidth=0.5, label="last sample")
      
      if i == 0: axes.legend(bbox_to_anchor=(0, 0), loc='upper left', fontsize=5)

  fig.subplots_adjust(bottom=0.05)
  cbar_ax = fig.add_axes([0.765, 0.1, 0.205, 0.03])
  ticks = np.linspace(0, im.get_clim()[1], 4)
  cbar = plt.colorbar(im, ticks=ticks, cax=cbar_ax, orientation='horizontal')
  cbar.ax.set_xticklabels(ticks.round(decimals=2))
  cbar.ax.tick_params(labelsize=5, length=2, pad=1)
  plt.tight_layout(pad=0.65)
  plt.show()

  return samp_idx

def run_sim(n_exp, n_samp, n_row, n_col, n_rew, s0, M,
            rew_size=1, userandom=True, idx=[2,8,14,27,34,49,52,65,79,87], rand_rew=False, reachable=None, 
            MFC=None, add_absorb_state=True,
            rho=None, beta=None, check_context_unit_norm=True, verbose=True):
  '''
  TCM-SR sample-based evaluation (most useful for rho=1, i.e., i.i.d. sampling)

  Params:
    n_exp: number of experiments/boards (int)
    n_samp: number of samples drawn per experiment (int)
    n_row: number of rows (int)
    n_col: number of columns (int)
    s0: starting position (same across experiments) (int tuple)
    M: successor representation of the specified gamma (numpy array)
    rew_size: reward magnitude (optional, number)
    userandom: use random reward placement (optional, bool)
    idx: pre-determined reward positions (optional, list)
    rand_rew: whether to use random reward magnitude for each placement (optional, bool)
    MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
          (optional, numpy array)
    add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
    beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
    check_context_unit_norm: if true, checks the norm of the updated context vector is 1 (optional, bool)
    verbose: if true, prints progress (% complete) (optional, bool)
  
  Return:
    vtrue: true values of s0 in each experiment (numpy array)
    vsamp: sample-based estimated value of s0 in each experiment (numpy array)
    (r_cnt, nr_cnt): tuples indicating the number of rewarding states sampled for the first 10, 100, and 1000 samples, and
                   the number of non-rewarding states sampled for the first 10, 100, and 1000 samples (numpy array)
    samped_row: sampled rows (numpy array)
  '''
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(n_exp)
  vsamp = np.zeros((n_exp,n_samp))
  samped_row = np.zeros((n_exp,n_samp), dtype=int)
  samped_col = np.zeros((n_exp,n_samp), dtype=int)
  r_cnt, nr_cnt = np.zeros((n_exp,3)), np.zeros((n_exp,3))

  for e in range(n_exp):
    if verbose and n_exp >= 10 and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
    maze = pachinko_sim.init_maze(n_row, n_col, n_rew, rew_size=rew_size, userandom=userandom, idx=idx, reachable=reachable)
    rvec = maze.flatten() # 1-step rewards
    if rand_rew: rvec = 1 + randn(rvec.size)
    if add_absorb_state: rvec = np.append(rvec, 0)

    state_idx = np.arange(0, n_row * n_col).reshape(maze.shape)    # state indices
    stim = np.identity(n_row * n_col + add_absorb_state)

    vtrue[e] = np.dot(M[state_idx[s0],:], rvec)                # Compute the actual value function (as v=Mr)

    c = stim[:,state_idx[s0]] # starting context vector (set to the starting state)

    samp_idx = np.negative(np.ones(n_samp, dtype=int))
    for i in range(n_samp):
      # define sampling distribution
      a = np.matmul(M.T,c)[:-add_absorb_state]
      P = a / np.sum(a)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(P))[0]
      samp_idx[i] = tmp[0]
      samped_row[e,i], samped_col[e,i] = np.unravel_index(samp_idx[i], (n_row, n_col))
      f = stim[:,samp_idx[i]]

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if check_context_unit_norm:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[samp_idx[samp_idx >= 0]]
    # count samples drawn from rewarding vs non-rewarding states
    r_cnt[e,0], nr_cnt[e,0] = sum(rewsamp[:10] > 0), sum(rewsamp[:10] == 0)
    r_cnt[e,1], nr_cnt[e,1] = sum(rewsamp[:100] > 0), sum(rewsamp[:100] == 0)
    r_cnt[e,2], nr_cnt[e,2] = sum(rewsamp[:1000] > 0), sum(rewsamp[:1000] == 0)
    rewsamp = np.append(rewsamp, np.zeros(n_samp - len(rewsamp)))
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[state_idx[s0],:]) 

  return vtrue, vsamp, (r_cnt, nr_cnt), samped_row

def run_sim_iw(n_exp, n_samp, n_row, n_col, maze, s0, M, M_ref,
            MFC=None, add_absorb_state=True, mod=False, mod_frac=0,
            rho=None, beta=None, check_context_unit_norm=True, verbose=True):
  '''
    TCM-SR sample-based evaluation with importance weighting
    NOTE: the game (maze) is provided (fixed) without the option to randomly construct a board

    Params:
    n_exp: number of experiments/boards (int)
    n_samp: number of samples drawn per experiment (int)
    n_row: number of rows (int)
    n_col: number of columns (int)
    maze: initialized Plinko board (numpy array)
    s0: starting position (same across experiments) (int tuple)
    M: successor representation of the specified gamma (numpy array)
    M_ref: reference successor representation (numpy array)
    MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
          (optional, numpy array)
    add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    mod: whether to use reward modulated learning or not (optional, bool)
    mod_frac: fraction of time a reward is expected to be sampled given reward modulation (optional, float)
    rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
    beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
    check_context_unit_norm: if true, checks the norm of the updated context vector is 1 (optional, bool)
    verbose: if true, prints progress (% complete) (optional, bool)
  
  Return:
    vtrue: true values of s0 in each experiment (numpy array)
    vref: reference true values of s0 in each experiment, corresponding to M_ref (numpy array)
    vsamp: sample-based estimated value of s0 in each experiment (numpy array)
    r_cnt: number of rewarding states sampled for the first 10, 100, and 1000 samples (numpy array)
    r_cnt: number of non-rewarding states sampled for the first 10, 100, and 1000 samples (numpy array)
    samped_row: sampled rows (numpy array)
  '''
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(n_exp)
  vref = np.zeros(n_exp)
  vsamp = np.zeros((n_exp,n_samp))
  samped_row = np.zeros((n_exp,n_samp), dtype=int)
  samped_col = np.zeros((n_exp,n_samp), dtype=int)
  r_cnt, nr_cnt = np.zeros((n_exp,3)), np.zeros((n_exp,3))

  idx = np.ravel_multi_index(np.where(maze > 0), (n_row, n_col))

  for e in range(n_exp):
    if verbose and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
    rvec = maze.flatten() # 1-step rewards
    if add_absorb_state: rvec = np.append(rvec, 0)

    state_idx = np.arange(0, n_row * n_col).reshape(maze.shape)    # state indices
    stim = np.identity(n_row * n_col + add_absorb_state)

    vtrue[e] = np.dot(M[state_idx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    vref[e] = np.dot(M_ref[state_idx[s0],:], rvec)       # Compute the actual value function (as v=Mr)

    c = stim[:,state_idx[s0]] # starting context vector (set to the starting state)

    samp_idx = np.negative(np.ones(n_samp, dtype=int))
    ps, qs = np.zeros(n_samp), np.zeros(n_samp)

    for i in range(n_samp):
      # define sampling distribution
      a = np.matmul(M.T,c)[:-add_absorb_state]
      Q = a / np.sum(a)
      assert np.abs(np.sum(Q)-1) < 1e-10, 'Q is not a valid probability distribution'

      # get reference distribution
      aref = np.matmul(M_ref.T,c)[:-add_absorb_state]
      P = aref / np.sum(aref)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(Q))[0]
      samp_idx[i] = tmp[0]

      if samp_idx[i] >= n_row * n_col: # sampled absorbing state:
        break
      else:
        samped_row[e,i], samped_col[e,i] = np.unravel_index(samp_idx[i], (n_row, n_col))
        f = stim[:,samp_idx[i]]
      
      if mod:
        if (random() < mod_frac):
          samp_idx[i] = idx[0]
          f = stim[:,samp_idx[i]]
      
      # store sampling probabilities
      ps[i], qs[i] = P[samp_idx[i]], Q[samp_idx[i]] * (1-mod_frac) if samp_idx[i] != idx[0] else Q[samp_idx[i]] * (1-mod_frac) + mod_frac

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if check_context_unit_norm:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[samp_idx[samp_idx >= 0]]
    rewsamp = np.append(rewsamp, np.zeros(n_samp - len(rewsamp)))
    # reweigh samples
    if mod: 
      rewsamp = np.divide(np.multiply(rewsamp, ps), qs)
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[state_idx[s0],:])
    # count samples drawn from rewarding vs non-rewarding states
    r_cnt[e,0], nr_cnt[e,0] = sum(rewsamp[:10] > 0), sum(rewsamp[:10] == 0)
    r_cnt[e,1], nr_cnt[e,1] = sum(rewsamp[:100] > 0), sum(rewsamp[:100] == 0)
    r_cnt[e,2], nr_cnt[e,2] = sum(rewsamp[:1000] > 0), sum(rewsamp[:1000] == 0)

  return vtrue, vref, vsamp, (r_cnt, nr_cnt), samped_row

def get_val_est_pctl(vsamp, n_samp, beta):
  percentiles = [2.5, 25, 50, 75, 97.5]
  mean = np.empty(n_samp)
  pctl = np.empty((len(percentiles), n_samp))
  if beta == 0:
    for s in range(n_samp):
      mean[s] = nanmean(nanmean(vsamp[:,:s+1],axis=1))
      pctl[:,s] = percentile(nanmean(vsamp[:,:s+1],axis=1), percentiles)
  else:
    for s in range(n_samp):
      mean[s] = nanmean(nansum(vsamp[:,:s+1],axis=1))
      pctl[:,s] = percentile(nansum(vsamp[:,:s+1],axis=1), percentiles)
  return mean, pctl
