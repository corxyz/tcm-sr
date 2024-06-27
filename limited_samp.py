import math
import numpy as np
import pylab as plt
from numpy.matlib import repmat
from numpy.random import randint
from matplotlib.gridspec import GridSpec
from numpy import percentile, nanmean
import util, pachinko_sim, choice

########################################
#  CONSTANTS & HYPERPARAMETERS
########################################

cm = 1/2.54  # centimeters in inches

n_row, n_col = 10, 9
min_rew, max_rew = 5, 24
rew_size = 1
rew_idx = [20,24,30,32,40]
s0 = (0,4)
add_absorb_state = True
extra_trans_on_sr = True

n_exp = 10                   # number of mazes
n_samp = 1000               # number of samples from each SR
rho, beta = 1, 0
alpha = 1
lamb = 0.7
gammas = [0.9, 0.9]
rew_mod_alphas = [None, 2]
lr_schedule_type = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.5}}

n_trial = 1              # number of trajectories for each maze
n_traj = 100
trial_to_plot = [1,2,3,4]
state_idx = np.arange(0, n_row * n_col).reshape((n_row, n_col))
n_state = n_row * n_col + add_absorb_state

########################################
#  HELPER FN
########################################

def getValueBiasPctl(vsamp, vtrue, percentiles=[2.5, 25, 50, 75, 97.5]):
  '''
    Compute statistics (mean, percentiles, variance, bias) of the 
    sample-based action value estimates over time

    Params:
      vsamp: sample-based estimated value of s0 in each experiment (numpy array, dim = n_exp x n_traj x n_samp)
      vtrue: true values of s0 in each experiment (numpy array)
      percentiles: list of percentile points (optional, list)
    
    Return:
      mean: average action value estimate as a function of samples (numpy array)
      pctl: percentiles of the action value estimate as a function of samples (numpy array)
      var: variance of the action value estimates as a function of samples (numpy array)
      bias: bias (avg estimate - truth) as a function of samples (numpy array)
  '''
  _, _, n_samp = vsamp.shape
  bias = vsamp - np.repeat(vtrue[:,:,np.newaxis], n_samp, axis=2)
  mean = np.zeros(n_samp)
  pctl = np.zeros((len(percentiles), n_samp))
  var = np.zeros(n_samp)
  for s in range(n_samp):
    mean[s] = nanmean(nanmean(bias[:,:,:s+1],axis=2))
    pctl[:,s] = percentile(nanmean(bias[:,:,:s+1],axis=2), percentiles)
    var[s] = np.var(nanmean(bias[:,:,:s+1],axis=2))
  return mean, pctl, var, bias

def plotValueEstLimitedSamps(n_samp, vtrues, vsamps, gammas=[], rew_mod_alphas=None, xabs_max=2):
  '''
    Plot the sample-based action value estimate over time for each intermediate SR

    Params:
      n_samp: number of samples drawn per experiment (int)
      vtrues: true action values of each experiment & gamma (numpy array, dim: n_gammas * n_pstops * n_exp)
      vsamps: sampled-based value estimates of each experiment & gamma 
              (numpy array, dim: n_gammas * n_pstops * n_exp * n_trial * n_samp)
      gammas: discount factors (optional, list)
      rew_mod_alphas: reward modulated learning rates (optional, None or list)
      xabs_max: maximum absolute value of the x-axis (optional, number)
  '''
  n_gamma, _, _, n_trial = vtrues.shape
  n_row, n_col = 2*n_gamma, n_trial
  fig = plt.figure(1, figsize=[n_col*3*1.3*cm, n_row*1.5*cm], dpi=300)
  gs = GridSpec(n_row, n_col*3, figure=fig, hspace=1.25, wspace=0.75)
  plt.clf()

  axes, axes2 = [], []
  ymax, ymax2 = 0, 6
  for i, gamma in enumerate(gammas):
    for k in range(n_trial):
      vsamp, vtrue = vsamps[i,:,:,k,:], vtrues[i,:,:,k]
      mean, pctl, _, bias = getValueBiasPctl(vsamp, vtrue)
      for j, nsamp in enumerate([10, 100, 1000]):
        if n_samp >= nsamp:
          ax = fig.add_subplot(gs[i*2,k*3+j])
          axes.append(ax)
          ax.set_title(r'$n=$' + str(nsamp), size=5, pad=2)
          tmp = nanmean(bias[:,:,:nsamp], axis=2).flatten()
          ax.hist(tmp, bins=np.arange(-xabs_max, xabs_max+0.1, 0.1), density=True)
          ymax = max(max(ax.get_ylim()), ymax)
          ax.set_xlabel('bias', fontdict={'fontsize':5}, labelpad=2)
          if j == 0 and k == 0: 
            ax.set_ylabel('density', fontdict={'fontsize':5}, labelpad=2)
            ax.tick_params(axis='both', which='major', labelsize=5, length=2, pad=1)
          else:
            ax.tick_params(labelleft=False, labelsize=5, length=2, pad=1)

      ax = fig.add_subplot(gs[i*2+1,k*3:(k+1)*3])
      alpha_mod = None
      if rew_mod_alphas: alpha_mod = rew_mod_alphas[i]
      if alpha_mod: 
        ax.set_title(r'$\alpha_{mod}=$' + str(alpha_mod), size=5, pad=0)
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
      ax.set_xlabel(r'$n$', fontdict={'fontsize':5}, labelpad=0)
      if k == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
      else: ax.tick_params(labelleft=False)
      ymax2 = max(max(ax.get_ylim()), ymax2)
    ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
  
  ymax2 = math.floor(ymax2)
  for ax in axes2:
    ax.set_ylim([-ymax2, ymax2])
  
  fig.subplots_adjust(right=0.8, top=0.92, bottom=0.12)
  plt.savefig('limited_samps.pdf', dpi=300)
  plt.show()

def plotSR(maze, traj, MTD):
  '''
    Plot the given successor representation and a trajectory overlaid on top

    Params:
      maze: Plinko board (numpy array)
      traj: a trajectory (list)
      MTD: TD-updated successor representation (numpy array)
  '''
  n_row, n_col = maze.shape
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[5*cm, 5*cm], dpi=300)
  # plot the board with traj and rewards marked
  for s in traj[:-add_absorb_state]:
    ax0.plot(s[1]+0.5, n_row-s[0]-0.5, 'kx', markersize=4, mew=0.5)
  rewR, rewC = np.where(maze > 0)
  for j, r in enumerate(rewR):
    c = rewC[j]
    ax0.plot(c+0.5, n_row-r-0.5, 'm*', markersize=4, mew=0.5)
  ax0.set_xticks(np.arange(0, n_col+1, 1))
  ax0.set_yticks(np.arange(0, n_row, 1))
  ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax0.grid()
  ax0.set_aspect('equal', adjustable='box')
  # plot the learned SR
  sr = MTD[4,:-add_absorb_state].reshape(maze.shape)
  # constrain all values to the prespecified range for plotting
  ax1.imshow(sr, vmin=0, vmax=.075, cmap="Greys")
  ax1.set_xticks(np.arange(-.5, n_col, 1))
  ax1.set_yticks(np.arange(-.5, n_row, 1))
  ax1.set_xticklabels(np.arange(0, n_col+1, 1))
  ax1.set_yticklabels(np.arange(0, n_row+1, 1))
  ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax1.grid()
  ax1.set_aspect('equal', adjustable='box')

  plt.tight_layout()
  plt.show()

########################################
#  MAIN ROUTINE
########################################

def main():
  MTDs = np.zeros((len(gammas), n_exp, n_traj, n_trial, n_state, n_state))
  vsamps = np.zeros((len(gammas), n_exp, n_traj, n_trial, n_samp))
  vtrues = np.zeros((len(gammas), n_exp, n_traj, n_trial))
  scheduler = schedulers[lr_schedule_type]
  for i, gamma in enumerate(gammas):
    # compute true transition matrix and SR
    T = pachinko_sim.init_trans(n_row, n_col, add_absorb_state=add_absorb_state)
    M = util.get_sr(T, gamma=gamma, extra_trans_on_sr=extra_trans_on_sr)
    for e in range(n_exp):
      # generate each maze
      maze = pachinko_sim.init_maze(n_row, n_col, randint(min_rew, max_rew), 
                                    rew_size=rew_size, userandom=True, reachable=s0, n_row_excl=2)
      for j in range(n_traj):
        # generate trajectories
        for t in range(n_trial):
          traj = pachinko_sim.get_rand_traj(maze, s0, add_absorb_state=add_absorb_state)
          # obtain intermediate SR
          if t == 0: 
            MTD = np.identity(n_state)
            lr = alpha
            alphaMod = rew_mod_alphas[i]
          else: 
            MTD = MTDs[i,e,j,t-1,:]
          MTDs[i,e,j,t,:], lr, alphaMod = choice.update_sr(t, lr, alphaMod, scheduler, maze, traj, MTD, gamma, lamb, add_absorb_state, extra_trans_on_sr)
          # estimate value via sampling
          vtrue, vsamp, _ = choice.samp_based_val_est(maze, n_samp, s0, M, MTDs[i,e,j,t,:], rew_mod_alphas[i])
          # record
          vtrues[i,e,j,t] = vtrue
          vsamps[i,e,j,t,:] = vsamp
  # plot value estimate convergence
  plotValueEstLimitedSamps(n_samp, vtrues, vsamps, 
                            gammas=gammas, rew_mod_alphas=rew_mod_alphas, xabs_max=5)

main()
