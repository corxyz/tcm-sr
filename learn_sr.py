import numpy as np
import pylab as plt
from scipy.stats import sem
from numpy.random import randint
from numpy.matlib import repmat
from numpy import nanmean
from matplotlib.gridspec import GridSpec
import bias_var, pachinko_sim, pstop_sim

########################################
#  CONSTANTS & HYPERPARAMETERS
########################################

cm = 1/2.54  # centimeters in inches

n_row, n_col = 10, 9
min_rew, max_rew = 20, 40
rew_size = 1
rew_idx = [20,24,30,32,40]
s0 = (0,4)
add_absorb_state = True
extra_trans_on_sr = True

n_exp = 100
n_samp = 1000
rho, beta = 1, 0
alpha = 0.01
lamb = 1
gammas = [0.9, 0.9]
rew_mod_alphas = [None, 0.5]
lr_schedule_type = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.001}}

n_trial = 10000
trial_to_plot = [1,2,3,4,n_trial]
mazes = [None] * (n_exp+1)
trajs = [[None] * n_trial for e in range(n_exp+1)]
state_idx = np.arange(0, n_row * n_col).reshape((n_row, n_col))

trueT = pachinko_sim.init_trans(n_row, n_col, add_absorb_state=add_absorb_state)
n_state = n_row * n_col + add_absorb_state
Ts = np.zeros((len(gammas), n_exp+1, len(trial_to_plot), n_state, n_state))

trueM_idx = 0  # index (w.r.t. the trial_to_plot array above) of iteratively learned SR to be used in evaluation 

########################################
#  HELPER FN
########################################

def generate_trajs():
  '''
    Generate random Plinko boards and sample trajectories
  '''
  print('Generating training data...', end='')
  for e in range(n_exp+1):
    if n_exp >= 10 and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%', end=' ')
    if e == 0:  # demo maze
      mazes[e] = pachinko_sim.init_maze(n_row, n_col, len(rew_idx), rew_size=rew_size, userandom=False, idx=rew_idx)
    else:
      mazes[e] = pachinko_sim.init_maze(n_row, n_col, randint(min_rew, max_rew), rew_size=rew_size, userandom=True, reachable=s0, n_row_excl=4)
    for t in range(n_trial):
      trajs[e][t] = pachinko_sim.get_rand_traj(mazes[e], s0, add_absorb_state=add_absorb_state)
  print()

def train_sr(scheduler):
  '''
    Incrementally learn the successor representation of each Plinko board

    Params:
      scheduler: scheduler: learning rate schedule (dictionary)
    
    Return:
      Ms: true successor representations of each Plinko board
      MTDs: empirically learned successor representations of each Plinko board
  '''
  n_state = n_row * n_col
  Ms = np.zeros((len(gammas), n_exp+1, len(trial_to_plot), n_state))
  MTDs = np.zeros((len(gammas), n_exp+1, len(trial_to_plot), n_state+add_absorb_state, n_state+add_absorb_state))

  print('Training...', end='')
  for i, gamma in enumerate(gammas):
    lr = alpha
    rew_mod_alpha = rew_mod_alphas[i]
    for j, maze in enumerate(mazes):
      M_init = np.identity(n_state+add_absorb_state)
      count = 0
      trajl = trajs[j]
      if n_exp >= 10 and (j+1) % (n_exp//10) == 0: print(str((j+1) * 100 // n_exp) + '%', end=' ')
      for k, traj in enumerate(trajl):
        # learning rate scheduler
        if lr_schedule_type == 'step':
          if (k+1) % scheduler['epochs'] == 0: # apply step decay
            lr *= scheduler['rate']
            if rew_mod_alpha: 
              rew_mod_alpha *= scheduler['rate']
        elif lr_schedule_type == 'exp':
          lr = alpha * np.exp(-scheduler['rate'] * k)
          if rew_mod_alpha: 
            rew_mod_alpha = rew_mod_alphas[i] * np.exp(-scheduler['rate'] * k)
        MTD = bias_var.get_sr_td(maze, traj, M_init, alpha=lr, gamma=gamma, lamb=lamb, add_absorb_state=add_absorb_state,
                                extra_trans_on_sr=extra_trans_on_sr, rew_thres=0, rew_mod_alpha=rew_mod_alpha)
        if k+1 in trial_to_plot:
          rvec = maze.flatten()
          print(i,j,k, np.dot(MTD[state_idx[s0],:-add_absorb_state],rvec))
          Ms[i,j,count,:] = MTD[state_idx[s0],:-add_absorb_state]
          MTDs[i,j,count,:,:] = MTD
          count += 1
        M_init = MTD
  print()
  return Ms, MTDs

def get_samped_val(MTDs):
  '''
    Compute the sample-based action value estimates of s0 using the empirically learned SRs

    Params:
      MTDs: empirically learned successor representations of each Plinko board (numpy array)
    
    Return:
      vtrues: true action values of each experiment & gamma (numpy array, dim: n_gammas * n_pstops * n_exp)
      vsamps: sampled-based value estimates of each experiment & gamma 
              (numpy array)
      samp_cnt_list: list of (r_cnt, nr_cnt) tuples, where
                    r_cnt is the number of rewarding states sampled for the first 10, 100, and 1000 samples,
                    nr_cnt is the number of non-rewarding states sampled for the first 10, 100, and 1000 samples
                    (numpy array)
  '''
  samp_cnt_list = [[] for i in range(len(gammas))]
  vsamps = np.zeros((len(gammas), n_exp, n_samp))
  vtrues = np.zeros((len(gammas), n_exp))
  # estimate sample-based value
  print('Testing...', end='')
  for i, gamma in enumerate(gammas):
    for j, maze in enumerate(mazes[1:]):
      if n_exp >= 10 and (j+1) % (n_exp//10) == 0: print(str((j+1) * 100 // n_exp) + '%', end=' ')
      M = MTDs[i,j+1,trueM_idx,:]
      M_ext = MTDs[0,j+1,trueM_idx,:]
      MTF_id = np.identity(M.shape[0])
      _, vtrue, vsamp, (r_cnt, nr_cnt), _ = bias_var.run_sim_iw(1, n_samp, n_row, n_col, maze,
                                                            s0, M, M_ext, MFC=MTF_id, add_absorb_state=add_absorb_state, 
                                                            rho=rho, beta=beta, verbose=False, 
                                                            mod=rew_mod_alphas[i], mod_frac=0)
      vtrues[i,j] = vtrue
      vsamps[i,j,:] = vsamp
      samp_cnt_list[i].append((r_cnt, nr_cnt))
  print()
  return vtrues, vsamps, samp_cnt_list

def plot_samp_rew_frac(samp_cnt_list, rew_mod_alphas):
  '''
    Plot sample distribution in terms of rewarding vs non-rewarding states

    Params:
      samp_cnt_list: list of (r_cnt, nr_cnt) tuples, where
                    r_cnt is the number of rewarding states sampled for the first 10, 100, and 1000 samples,
                    nr_cnt is the number of non-rewarding states sampled for the first 10, 100, and 1000 samples
                    (numpy array)
      rew_mod_alphas: reward modulated learning rates (list)
  '''
  _, axes = plt.subplots(len(rew_mod_alphas), 3, figsize=[6.6*cm, 4.4*cm], dpi=300)
  for i, rew_mod_alpha in enumerate(rew_mod_alphas):
    l = samp_cnt_list[i]
    r_cnt_avg, nr_cnt_avg = np.array([r_cnt.mean(axis=0) for (r_cnt, _) in l]), np.array([nr_cnt.mean(axis=0) for (_, nr_cnt) in l])
    total = r_cnt_avg + nr_cnt_avg
    r_cnt_avg, nr_cnt_avg = r_cnt_avg / total, nr_cnt_avg / total
    height = np.array([r_cnt_avg.mean(axis=0), nr_cnt_avg.mean(axis=0)])
    errs = np.array([sem(r_cnt_avg), sem(nr_cnt_avg)])
    for j in range(height.shape[1]):
      axes[i,j].bar([0,1], height[:,j], tick_label=['r.', 'n.r.'], yerr=errs[:,j],
                    error_kw=dict(lw=1), color=['m', 'gray'])
      axes[i,j].set_ylim([0, 1])
      axes[i,j].set_ylabel('average reward\nfraction', fontdict={'fontsize':5}, labelpad=1)
      axes[i,j].tick_params(labelsize=5, length=2, pad=1)
      plt.setp(axes[i,j].spines.values(), linewidth=0.5)
      if rew_mod_alpha:
        axes[i,j].set_title(r'$\alpha_{mod}=$' + str(rew_mod_alpha), size=5, pad=2)
  plt.tight_layout(pad=0.25)
  plt.show()

def plot_val_est_learned_sr(n_samp, vtrues, vsamps, beta, gammas=[], rew_mod_alphas=None, xabs_max=5):
  '''
    Plot the sample-based action value estimate over time

    Params:
      n_samp: number of samples drawn per experiment (int)
      vtrues: true action values of each experiment & gamma (numpy array, dim: n_gammas * n_pstops * n_exp)
      vsamps: sampled-based value estimates of each experiment & gamma 
              (numpy array, dim: n_gammas * n_pstops * n_exp * n_trial * n_samp)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (float)
      gammas: discount factors (optional, list)
      rew_mod_alphas: reward modulated learning rates (optional, None or list)
      xabs_max: maximum absolute value of the x-axis (optional, number)
  '''
  n_row, n_col = 2, len(gammas)
  fig = plt.figure(1, figsize=[n_col*3*2*cm, n_row*2.4*cm], dpi=300)
  gs = GridSpec(n_row, n_col*3, figure=fig, hspace=0.8)
  plt.clf()

  axes, axes2 = [], []
  ymax = 0
  for i, gamma in enumerate(gammas):
    vsamp, vtrue = vsamps[i], vtrues[i]
    mean, pctl, bias = pstop_sim.get_val_est_pctl(vsamp, vtrue, beta)
    for j, nsamp in enumerate([10, 100, 1000]):
      if n_samp >= nsamp:
        ax = fig.add_subplot(gs[0,i*3+j])
        axes.append(ax)
        ax.set_title(r'$N=$' + str(nsamp), size=5, pad=2)
        tmp = nanmean(bias[:,:nsamp], axis=1)
        ax.hist(tmp, bins=np.arange(-xabs_max, xabs_max+0.01, 0.01), density=True)
        ymax = max(max(ax.get_ylim()), ymax)
        ax.set_xlabel('bias', fontdict={'fontsize':5}, labelpad=2)
        if j == 0 and i == 0: 
          ax.set_ylabel('density', fontdict={'fontsize':5}, labelpad=2)
          ax.tick_params(axis='both', which='major', labelsize=5, length=2, pad=1)
        else:
          ax.tick_params(labelleft=False, labelsize=5, length=2, pad=1)

    ax = fig.add_subplot(gs[-1,i*3:(i+1)*3])
    alpha_mod = None
    if rew_mod_alphas: alpha_mod = rew_mod_alphas[i]
    if alpha_mod: 
      ax.set_title(r'$\alpha_{mod}=$' + str(alpha_mod), size=7, pad=2)
    axes2.append(ax)
    colors = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,0.1,0.1],[0.3,0.3,0.3],[0.5,0.5,0.5]]
    ls = [':','--','-','--',':']
    labels = ['95% CI', '50% CI', 'Median', '', '']
    lines1 = ax.semilogx(repmat(np.arange(0,n_samp),5,1).T, pctl.T, linewidth=0.7)
    for l, line in enumerate(ax.get_lines()):
      line.set_linestyle(ls[l])
      line.set_color(colors[l])
      if (len(labels[l]) > 0): line.set_label(labels[l])
    lines2 = ax.semilogx(np.arange(0,n_samp), mean, color=[0,0,0.8], linewidth=1, linestyle='-', label='Mean')
    ax.grid(True)
    ax.tick_params(labelsize=5, length=2, pad=0)
    ax.set_xlabel(r'$N$', fontdict={'fontsize':5}, labelpad=0)
    if i == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
    else: ax.tick_params(labelleft=False)
  ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
    ax.set_xticks([-xabs_max,0,xabs_max])
  
  for ax in axes2:
    ax.set_ylim([-xabs_max, xabs_max])
    ax.set_yticks([-xabs_max,0,xabs_max])
  
  fig.subplots_adjust(right=0.8, top=0.92, bottom=0.12)
  plt.savefig('f5c.pdf', dpi=300)
  plt.show()

def plot_learning_ex(MTDs):
  '''
    Plot the intermediate learned SRs (trial number specified in trial_to_plot)

    Params:
      MTDs: empirically learned successor representations of each Plinko board (numpy array)
  '''
  # plot boards & learned SR over time
  vmin, vmax = [0] * len(gammas), [None] * len(gammas)
  vmax_alt = 0.01
  assert(len(gammas) == len(rew_mod_alphas))
  for i, rma in enumerate(rew_mod_alphas):
    if rma: vmax[i] = .5
    else: vmax[i] = .5

  demo_idx = 0
  s0_idx = np.ravel_multi_index(s0, (n_row, n_col))
  maze = mazes[demo_idx]
  tjs = trajs[demo_idx]

  fig = plt.figure(1, figsize=[18*cm, 5*cm], dpi=300)
  ims = []
  for ti, t in enumerate(trial_to_plot):
    traj = tjs[t-1]
    for i, gamma in enumerate(gammas):
      rew_mod_alpha = rew_mod_alphas[i]
      # plot the board with traj and rewards marked
      ax0 = plt.subplot(len(gammas),len(trial_to_plot)*2,i*len(trial_to_plot)*2+ti*2+1)
      for s in traj[:-add_absorb_state]:
        ax0.plot(s[1]+0.5, n_row-s[0]-0.5, 'kx', markersize=4, mew=0.5)
      rewR, rewC = np.where(maze > 0)
      for j, r in enumerate(rewR):
        c = rewC[j]
        if i == 0 and j == 0: ax0.plot(c+0.5, n_row-r-0.5, 'm*', markersize=4, mew=0.5, label='Reward')
        else: ax0.plot(c+0.5, n_row-r-0.5, 'm*', markersize=4, mew=0.5)
      if t < 5: ax0.set_title('Trial {}'.format(t), size=5, pad=2)
      ax0.set_xticks(np.arange(0, n_col+1, 1))
      ax0.set_yticks(np.arange(0, n_row, 1))
      ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
      ax0.grid()
      ax0.set_aspect('equal', adjustable='box')
      # plot the learned SR
      ax1 = plt.subplot(len(gammas),len(trial_to_plot)*2,i*len(trial_to_plot)*2+ti*2+2)
      sr = MTDs[i,demo_idx,ti,s0_idx,:-add_absorb_state].reshape(maze.shape)
      # constrain all values to the prespecified range for plotting
      if t < 10:
        sr = np.clip(sr, vmin[i], vmax_alt)
        im = ax1.imshow(sr, vmin=vmin[i], vmax=vmax_alt, cmap="Greys")
      else:
        sr = np.clip(sr, vmin[i], vmax[i])
        im = ax1.imshow(sr, vmin=vmin[i], vmax=vmax[i], cmap="Greys")
      if ti == 0 or ti == len(trial_to_plot)-1: ims.append(im)
      modulateStr = r'$\alpha_{mod}=$' + str(rew_mod_alpha) if rew_mod_alpha else r'$\alpha=$' + str(alpha)
      ax1.set_title(modulateStr, size=5, pad=2)
      ax1.set_xticks(np.arange(-.5, n_col, 1))
      ax1.set_yticks(np.arange(-.5, n_row, 1))
      ax1.set_xticklabels(np.arange(0, n_col+1, 1))
      ax1.set_yticklabels(np.arange(0, n_row+1, 1))
      ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
      ax1.grid()
      ax1.set_aspect('equal', adjustable='box')

  plt.tight_layout()
  fig.subplots_adjust(left=0.11, right=0.9)
  mid = len(ims) // 2
  offset = 1/len(ims[mid:]) - 0.05
  # plot the left colorbars
  for i, im in enumerate(ims[:mid]):
    cbar_ax = fig.add_axes([0.04, offset*(len(ims[mid:])-i-1)+0.15, 0.01, 0.25])
    vmx = vmax_alt
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[vmin[i], vmx])
    ticklabels = [r'$\leq $' + str(vmin[i]), r'$\geq $' + str(vmx)]
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=5, pad=2)
  # plot the right colorbars
  for i, im in enumerate(ims[mid:]):
    cbar_ax = fig.add_axes([0.925, offset*(len(ims[mid:])-i-1)+0.15, 0.01, 0.25])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[vmin[i], vmax[i]])
    ticklabels = [r'$\leq $' + str(vmin[i]), r'$\geq $' + str(vmax[i])]
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=5, pad=2)
  plt.savefig('f5-traj.pdf', dpi=300)
  plt.show()

########################################
#  MAIN ROUTINE
########################################

def main():
  # generate trajectories
  generate_trajs()
  # learn SR
  scheduler = schedulers[lr_schedule_type]
  Ms, MTDs = train_sr(scheduler)
  vtrues, vsamps, samp_cnt_list = get_samped_val(MTDs)
  plot_samp_rew_frac(samp_cnt_list, rew_mod_alphas)
  # plot bias-variance tradeoff
  plot_val_est_learned_sr(n_samp, vtrues, vsamps, beta, 
                        gammas=gammas, rew_mod_alphas=rew_mod_alphas, xabs_max=.2)
  # plot a few intermediate SRs
  plot_learning_ex(MTDs)

main()
