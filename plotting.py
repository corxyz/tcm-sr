import math
import numpy as np
import pylab as plt
import seaborn as sns
from numpy.random import rand
from numpy.matlib import repmat
from numpy import nanmean, nansum
from matplotlib.gridspec import GridSpec
import pachinko_sim, bias_var, pstopSim

########################################
#  CONSTANT
########################################

cm = 1/2.54

########################################
#  HELPER FN
########################################

def plot_maze(maze):
  '''
    Plot Plinko board

    Params:
      maze: Plinko board (numpy array)
  '''
  n_row, n_col = maze.shape
  plt.figure(figsize=[2.2*cm, 3*cm], dpi=300)
  # plot the board with starting position marked
  plt.xticks(np.arange(-.5, n_col, 1))
  plt.yticks(np.arange(-.5, n_row, 1))
  plt.tick_params(length=0, labelbottom=False, labelleft=False)   
  plt.grid()
  plt.axis('scaled')
  plt.show()

def plot_T_SR(n_row, n_col, state_idx, s0, T, M, add_absorb_state=False):
  '''
    Plot the given one-step transition matrix and the successor representation

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      state_idx: state index (list)
      s0: starting position (same across experiments) (int tuple)
      T: one-step transition matrix (numpy array)
      M: successor representation of the specified gamma (numpy array)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
  '''
  powerT = T.copy()
  fig = plt.figure(1, figsize=[15*cm, 5*cm], dpi=300)
  # plot transitions matrices
  for i in range(1,n_row):
    ax0 = plt.subplot(1,n_row,i)
    t = np.reshape(powerT[state_idx[s0],:-add_absorb_state], (n_row, n_col))
    im = ax0.imshow(t, vmin=0, vmax=0.5, cmap="Greys")
    ax0.set_title(r'$T^{}$'.format(i), size=5)
    ax0.set_xticks(np.arange(-.5, n_col, 1))
    ax0.set_yticks(np.arange(-.5, n_row, 1))
    ax0.set_xticklabels(np.arange(0, n_col+1, 1))
    ax0.set_yticklabels(np.arange(0, n_row+1, 1))
    ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
    ax0.grid()
    ax0.set_aspect('equal', adjustable='box')
    powerT = np.matmul(powerT, T)
  cbar_ax = fig.add_axes([0.04, 0.36, 0.01, 0.275])
  cbar = plt.colorbar(im, cax=cbar_ax, ticks=[0, 0.5])
  cbar.ax.tick_params(labelsize=5, pad=2)
  
  # plot SR
  ax1 = plt.subplot(1,n_row, n_row)
  sr = np.reshape(M[state_idx[s0],:-add_absorb_state], (n_row, n_col))
  im = ax1.imshow(sr, cmap="Greys")
  ax1.set_title(r'$M$', size=5)
  ax1.set_xticks(np.arange(-.5, n_col, 1))
  ax1.set_yticks(np.arange(-.5, n_row, 1))
  ax1.set_xticklabels(np.arange(0, n_col+1, 1))
  ax1.set_yticklabels(np.arange(0, n_row+1, 1))
  ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax1.grid()
  ax1.set_aspect('equal', adjustable='box')

  plt.show()

def make_square_axes(ax):
  '''
    Make an axes square in screen units (call after plotting).
  '''
  ax.set_aspect(1 / ax.get_data_ratio())

def plot_recency(n_exp, n_row, n_col, s0, M, MFC=None, rho=None, beta=None, add_absorb_state=False):
  '''
    Plot the recency curve over rows, averaged across experiments (i.i.d. sampling)

    Params:
      n_exp: number of experiments/boards (int)
      n_row: number of rows (int)
      n_col: number of columns (int)
      s0: starting position (same across experiments) (int tuple)
      M: successor representation (numpy array)
      MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
          (optional, numpy array)
      rho: TCM param, specifies how much of the previous context vector to retain (optional, float)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (optional, float)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
  '''
  recalls = np.zeros(n_exp)
  for e in range(n_exp):
    state_idx = np.arange(0, n_row * n_col).reshape((n_row, n_col))    # state indices
    stim = np.identity(n_row * n_col + add_absorb_state)
    c = stim[:,state_idx[s0]] # starting context vector (set to the starting state)

    curRow, curCol = s0
    for i in range(n_row-s0[0]-1):
      # proceed by one step
      curRow, curCol = pachinko_sim.rand_step_plinko(n_row, n_col, curRow, curCol)
      f = stim[:,state_idx[curRow, curCol]]
      # incoming context vector
      cIN = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN/np.linalg.norm(cIN)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'
      # update context
      c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # recall
    a = np.matmul(M.T,c)
    Q = a / np.sum(a)
    tmp = np.where(rand() <= np.cumsum(Q))[0]
    samp = tmp[0]
    recalls[e] = np.unravel_index(samp, (n_row, n_col))[0]

  plt.figure(1,figsize=[3.75*cm, 3.75*cm], dpi=300)
  plt.clf()
  p = np.zeros(n_row)
  for i in range(n_row):
    p[i] = (recalls == i).sum()/n_exp
  plt.plot(np.arange(1,n_row+1), p, 'ko-', markersize=2.5, linewidth=0.7)
  plt.xticks(np.arange(0,n_row+1,2), fontsize=5)
  plt.yticks(np.linspace(0, 0.8, 5), fontsize=5)
  plt.xlabel('Row', fontsize=5)
  plt.ylabel('Probability', fontsize=5)
  ax = plt.gca()
  ax.tick_params(length=2, direction='in')
  make_square_axes(ax)
  plt.tight_layout()
  plt.show()

def plot_crp_iid(samped_row, n_row, maxDist=9, omitZero=True):
  '''
    Plot the CRP curve over rows, averaged across experiments (i.i.d. sampling)

    Params:
      samped_row: sampled rows (numpy array)
      n_row: number of rows (int)
      maxDist: maximum relative spatial distance to plot (optional, int)
      omitZero: whether to plot transition to a state on the same row or not (optional, bool)
  '''
  n_exp, n_samp = samped_row.shape
  x = np.arange(-maxDist, maxDist+1)  # samples could come from row 1 to n_row-1 and we restrict it to the range of maxDist
  allCount = np.zeros(n_row * 2 - 1)
  count = np.zeros((n_exp, maxDist * 2 + 1))  # compute CRP curve for each experiment
  # accumulate counts of relative positions of consecutive samples
  for e in range(n_exp):
    samps = samped_row[e,:]
    for i in range(n_samp):
      if samps[i] < 0: break    # bottom reached
      if i == 0:  # i.i.d. samples or the first sample
        relPos = samps[i] if samps[i] > 0 else maxDist + 1
      else:
        relPos = samps[i] - samps[i-1]
      if abs(relPos) < n_row: allCount[relPos + n_row - 1] += 1
      if omitZero and relPos == 0 or abs(relPos) > maxDist: continue
      count[e, relPos + maxDist] += 1

  # compute density, mean, and standard error across experiments
  sum_of_rows = count.sum(axis=1)
  y = count / sum_of_rows[:, np.newaxis]
  if omitZero: y[:, maxDist] = np.nan
  mean = np.nanmean(y, axis=0)
  # plot graph
  plt.figure(1,figsize=[3.5*cm, 3.5*cm], dpi=300)
  plt.clf()
  plt.plot(x, mean, 'ko-', markersize=2, linewidth=0.7)
  plt.xticks([-9,-5,-1,1,5,9], fontsize=5)
  plt.yticks(np.linspace(0, np.round(np.nanmax(mean),1), 4), fontsize=5)
  plt.xlabel('d', fontsize=5)
  plt.ylabel('CRP', fontsize=5)
  ax = plt.gca()
  ax.tick_params(length=2, direction='in')
  make_square_axes(ax)
  plt.tight_layout()
  plt.show()

def plot_crp_noniid(samped_row, n_row, maxDist=5, ymax=None, omitZero=True):
  '''
    Plot the CRP curve over rows, averaged across experiments (non-i.i.d. sampling)

    Params:
      samped_row: sampled rows (numpy array)
      n_row: number of rows (int)
      maxDist: maximum relative spatial distance to plot (optional, int)
      ymax: maximum value plotted on the y-axis (optional, None or number)
      omitZero: whether to plot transition to a state on the same row or not (optional, bool)
  '''
  n_exp, n_trial, n_samp = samped_row.shape
  x = np.arange(-maxDist, maxDist+1)  # samples could come from row 0 to n_row-1 and we restrict it to the range of maxDist
  allCount = np.zeros((n_exp, n_row * 2 - 1))
  count = np.zeros((n_exp, n_trial, maxDist * 2 + 1))  # compute CRP curve for each experiment and each stopping probability
  # accumulate counts of relative positions of consecutive samples
  for e in range(n_exp):
    for t in range(n_trial):
      samps = samped_row[e,t,:]
      for i in range(n_samp):
        if samps[i] < 0: break
        if i == 0: 
          relPos = samps[i] if samps[i] > 0 else maxDist + 1
        else: 
          relPos = samps[i] - samps[i-1]
        if abs(relPos) < n_row: allCount[e, relPos + n_row - 1] += 1
        if omitZero and relPos == 0 or abs(relPos) > maxDist: continue
        count[e, t, relPos + maxDist] += 1
  
  # compute density, mean, and standard error across experiments for each stopping probability
  sum_of_rows = count.sum(axis=2)
  y = count / sum_of_rows[:, :, np.newaxis]
  if omitZero: y[:, :, maxDist] = np.nan
  mean = np.nanmean(y, axis=(0,1))
  # plot graph
  fig, ax = plt.subplots(figsize=[1.5*cm, 3.5*cm], dpi=300)
  ax.plot(mean, x, 'ko-', markersize=2.5, linewidth=0.7)
  ax.tick_params(axis='both', which='major', length=2, direction='in',labelsize=5)
  ax.set_yticks([-maxDist,-1,1,maxDist])
  if not ymax: mean_max = np.round(np.nanmax(mean),1)
  ax.set_xticks(np.linspace(0, mean_max, 3))
  ax.set_ylabel(r'$\Delta$' + 'row', fontdict={'fontsize':5}, labelpad=2, rotation=90)
  ax.set_xlabel('Probability', fontdict={'fontsize':5}, labelpad=2)
  ax.invert_yaxis()
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  fig.subplots_adjust(left=0.3, bottom=0.25)
  plt.show()

def plot_sample_dist_iid(n_row, n_exp, samped_rowList, gammas=[], rew_mod_alphas=None):
  '''
    Plot the empirical sampling distribution over rows along with the expected (theoretical) result 
    Specifically for plotting i.i.d. sampling results (i.e., beta = 0)

    Params:
      n_row: number of rows (int)
      n_exp: number of experiments/boards (int)
      samped_rowList: sampled rows in each experiment (list)
      gammas: discount factors (optional, list)
      rew_mod_alphas: reward modulated learning rates of each gamma (optional, None or list)
  '''
  assert(len(samped_rowList) == len(gammas))
  _, axes = plt.subplots(len(gammas), 1, figsize=[3*cm,2.8*len(gammas)*cm], dpi=300)

  # plot the sampling distributions of different gammas
  y = np.arange(0, n_row, 1)
  c = 'g'
  for i, gamma in enumerate(gammas):
    ax = axes[i]
    ax.invert_yaxis()
    d = samped_rowList[i].flatten()
    d = d[d >= 0]
    cnt = np.array([(d == r).sum()/float(n_exp) if r > 0 else 0 for r in y])
    psamp = cnt / cnt.sum()
    psamp[0] = np.nan
    theo = np.array([(1-gamma) * gamma ** (r-1) if r > 0 else 0 for r in y])
    theo = theo / theo.sum()
    theo[0] = np.nan
    
    alpha_mod = None
    if rew_mod_alphas: alpha_mod = rew_mod_alphas[i]
    if alpha_mod:
      ax.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.5,
                label='empirical (' + r'$\alpha_{mod}=$' + str(alpha_mod) + ')')
    else:
      ax.plot(theo, y, c[i%len(c)] + '-', linewidth=0.5, label='expected')
      ax.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.8, label='empirical')
    ax.set_xticks(np.linspace(0, 1, num=5))
    ax.set_yticks(np.arange(0, n_row, 1))
    ax.set_xlabel(r'$P(i_t)$', fontdict={'fontsize':5}, labelpad=2)
    ax.set_ylabel('Row number', fontdict={'fontsize':5}, labelpad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)

  plt.tight_layout(pad=0.25)
  plt.show()

def plot_sample_dist_noniid(samped_row, n_row, n_exp, pstops, eGammas, legend_in=False):
  '''
    Plot the empirical sampling distribution over rows along with the expected (theoretical) result 
    Specifically for plotting i.i.d. sampling results (i.e., beta = 0)

    Params:
      samped_row: sampled rows in each experiment (list)
      n_row: number of rows (int)
      n_exp: number of experiments/boards (int)
      pstops: interruption probabilities (list)
      eGammas: effective discount factors, one for each pstop (list)
      legend_in: if true, plot the legend inside the frame (optional, bool)
  '''
  figsize = [3.5*cm,3*cm] if legend_in else [5*cm,3*cm]
  plt.figure(figsize=figsize, dpi=300)

  # plot the sampling distributions of different effective gammas
  y = np.arange(0, n_row, 1)
  c = ['r','b','g']
  plt.gca().invert_yaxis()
  for i, pstop in enumerate(pstops):
    gamma = eGammas[i]
    d = samped_row[i,:,:,:].flatten()
    d = d[d >= 0]
    cnt = np.array([(d == r).sum()/float(n_exp) for r in y])
    cnt[0] = np.nan
    psamp = cnt / np.nansum(cnt)
    theo = np.array([(1-gamma) * gamma ** (r-1) if r > 0 else np.nan for r in y])
    theo = theo / np.nansum(theo)
    label = r'$\widetilde{\gamma}=$' + str(gamma) if gamma < 0.99 else r'$\widetilde{\gamma}\to 1$'
    plt.plot(theo, y, c[i%len(c)] + '-', linewidth=0.5, label=label)
    plt.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.8)
  plt.xticks(np.linspace(0, 1, num=5), fontsize=5)
  plt.yticks(np.arange(0, n_row, 1), fontsize=5)
  plt.xlabel(r'$P(i_t)$', size=5, labelpad=1)
  plt.ylabel('Row number ', size=5, labelpad=1)

  if legend_in: 
    plt.legend(loc='lower right', fontsize=5)
  else: 
    plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left', fontsize=5)
  plt.tight_layout(pad=0.6)
  plt.show()

def plot_val_est_iid(nSamples, vtrues, vsamps, beta, gammas=None, rew_mod_alphas=None, xabs_max=5):
  nRows, nCols = 2 * len(gammas), 1
  fig = plt.figure(1, figsize=[nCols*3*1.5*cm, nRows*1.5*cm], dpi=300)
  plt.clf()
  gs = GridSpec(nRows, nCols*3, figure=fig, hspace=1.1)

  axes, axes2 = [], []
  ymax, ymax2 = 0, 0
  for i, gamma in enumerate(gammas):
    vsamp, vtrue = vsamps[i], vtrues[i]
    mean, pctl, bias = bias_var.get_val_est_pctl(vsamp, nSamples, beta)
    for j, nsamp in enumerate([10, 100, 1000]):
      if nSamples >= nsamp:
        ax = fig.add_subplot(gs[i*2,j])
        axes.append(ax)
        ax.set_title(r'$N=$' + str(nsamp), size=5, pad=2)
        if nsamp < 100: tmp = np.concatenate([nanmean(bias[:,k:k+nsamp], axis=1) for k in range(0, nSamples-nsamp+1, nsamp)])
        else: tmp = nanmean(bias[:,:nsamp], axis=1)
        ax.hist(tmp, bins=np.arange(-xabs_max, xabs_max+0.1, 0.1), density=True)
        ymax = max(max(ax.get_ylim()), ymax)
        ax.set_xlabel('bias', fontdict={'fontsize':5}, labelpad=2)
        if j == 0: 
          ax.set_ylabel('density', fontdict={'fontsize':5}, labelpad=2)
          ax.tick_params(axis='both', which='major', labelsize=5, length=2, pad=1)
        else:
          ax.tick_params(labelleft=False, labelsize=5, length=2, pad=1)

    ax = fig.add_subplot(gs[i*2+1,:])
    alpha_mod = None
    if rew_mod_alphas: alpha_mod = rew_mod_alphas[i]
    if alpha_mod: 
      ax.set_title(r'$\alpha_{mod}=$' + str(alpha_mod), size=7, pad=5)
    axes2.append(ax)
    colors = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,0.1,0.1],[0.3,0.3,0.3],[0.5,0.5,0.5]]
    ls = [':','--','-','--',':']
    labels = ['95% CI', '50% CI', 'Median', '', '']
    lines1 = ax.semilogx(repmat(np.arange(0,nSamples),5,1).T, pctl.T, linewidth=0.7)
    for l, line in enumerate(ax.get_lines()):
      line.set_linestyle(ls[l])
      line.set_color(colors[l])
      if (len(labels[l]) > 0): line.set_label(labels[l])
    lines2 = ax.semilogx(np.arange(0,nSamples), mean, color=[0,0,0.8], linewidth=1, linestyle='-', label='Mean')
    ax.grid(True)
    ax.tick_params(labelsize=5, length=2, pad=0)
    ax.set_xlabel(r'$N$', fontdict={'fontsize':5}, labelpad=0)
    ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5}, labelpad=2)
    ymax2 = max(max(ax.get_ylim()), ymax2)
    # ax.set_ylim([mean[-1]-5,mean[-1]+5])
    ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
  
  ymax2 = math.floor(ymax2)
  for ax in axes2:
    ax.set_ylim([-ymax2, ymax2])
  
  fig.subplots_adjust(left=0.19, right=0.65, hspace=2)
  plt.tight_layout(pad=0.5)
  plt.show()

def plot_val_est_noniid(n_samp, n_trial, vtruesList, vsampsList, 
                 beta, gammas, eGammas, xabs_max=5, savename='biasvar'):
  '''
    Plot and save the sample-based action value estimate over time

    Params:
      n_samp: number of samples drawn per experiment (int)
      n_trial: number of trials (rollouts) per experiment (int)
      vtruesList: true action values of each experiment & gamma 
                  (numpy array, dim = n_gammas * n_pstops * n_exp)
      vsampsList: sampled-based value estimates of each experiment & gamma 
                  (numpy array, dim: n_gammas * n_pstops * n_exp * n_trial * n_samp)
      beta: TCM param, specifies how much of the new incoming context vector to incorporate (float)
      gammas: discount factors (list)
      eGammas: effective discount factors (list, dim = n_gammas * n_pstops)
      xabs_max: max absolute value to plot on the x-axis (non-negative number)
      savename: file name to save the plot with (optional, string)
  '''
  n_row, n_col = 2*len(gammas), len(eGammas[0])
  fig = plt.figure(1, figsize=[n_col*3*2*cm, n_row*2.4*cm], dpi=300)
  gs = GridSpec(n_row, n_col*3, figure=fig, hspace=0.9)
  plt.clf()

  axes, axes2 = [], []
  ymax, ymax2 = 0, 6
  for k, gamma in enumerate(gammas):
    vsamps, vtrues = vsampsList[k], vtruesList[k]
    for i, egamma in enumerate(eGammas[k]):
      vsamp, vtrue = vsamps[i], vtrues[i]
      vsamp[vsamp == 0] = np.nan
      mean, pctl, bias = pstopSim.get_val_est_pctl(vsamp, vtrue, beta)
      # plot bias density plots
      for j, n in enumerate([10, 100, 1000]):
        if beta != 0 and n_trial*n_samp >= n or beta == 0 and n_samp >= n:
          ax = fig.add_subplot(gs[2*k,i*3+j])
          axes.append(ax)
          ax.set_title(r'$n=$' + str(n), size=5, pad=2)
          ax.hist(bias[:,n-1], bins=np.arange(-xabs_max, xabs_max+0.1, 0.1), density=True)
          # xabs_max = max(abs(max(ax.get_xlim(), key=abs)), xabs_max)
          ymax = max(max(ax.get_ylim()), ymax)
          ax.set_xlabel('bias', fontdict={'fontsize':5}, labelpad=2)
          ax.xaxis.labelpad = 2
          if j == 0 and i == 0:
            ax.set_ylabel('density', fontdict={'fontsize':5}, labelpad=2)
            ax.tick_params(axis='both', which='major', labelsize=5, length=2, pad=1)
          else:
            ax.tick_params(labelleft=False, labelsize=5, length=2, pad=1)
      # plot estimated value plot
      ax = fig.add_subplot(gs[2*k+1,i*3:(i+1)*3])
      ax.set_title(r'$\widetilde{\gamma}=$' + str(egamma), size=5, pad=2)
      axes2.append(ax)
      colors = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,0.1,0.1],[0.3,0.3,0.3],[0.5,0.5,0.5]]
      ls = [':','--','-','--',':']
      labels = ['95% CI', '50% CI', 'Median', '', '']
      lines1 = ax.semilogx(repmat(np.arange(pctl.shape[1]),5,1).T, pctl.T, linewidth=0.7)
      for l, line in enumerate(ax.get_lines()):
        line.set_linestyle(ls[l])
        line.set_color(colors[l])
        if (len(labels[l]) > 0): line.set_label(labels[l])
      lines2 = ax.semilogx(np.arange(mean.shape[0]), mean, color=[0,0,0.8], linewidth=1, linestyle='-', label='Mean')
      ax.grid(True)
      ax.tick_params(labelsize=5, length=2, pad=0)
      ax.set_xlabel(r'$n$', fontdict={'fontsize':5}, labelpad=0)
      if i == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
      else: ax.tick_params(labelleft=False)
      ymax2 = max(max(ax.get_ylim()), ymax2)
    ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
    ax.set_xticks([-xabs_max,0,xabs_max])
  
  # ymax2 = math.floor(ymax2)
  for ax in axes2:
    # ax.set_ylim([-ymax2, ymax2])
    ax.set_ylim([-xabs_max, xabs_max])
    ax.set_yticks([-xabs_max,0,xabs_max])

  fig.subplots_adjust(right=0.8, top=0.92, bottom=0.12)
  plt.savefig(savename + '.pdf', dpi=300)
  plt.show()

def plot_est_err_dist(veffectList, vsamp_list, eGammas, is_sample=True, title='Density plot'):
  '''
    Plot the error distribution of value estimates against the true value

    Params:
      veffectList: true value of s0 w.r.t. the effective gamma of each experiment and gamma 
                   (numpy array, dim = n_gammas * n_pstops * n_exp)
      vsamp_list: sampled-based value estimates of each experiment & gamma 
                 (numpy array, dim = n_gammas * n_pstops * n_exp * n_trial * n_samp)
      eGammas: effective discount factors (list, dim = n_gammas * n_pstops)
      is_sample: if true, the values in vsamp_list are individual sampled rewards (optional, bool)
      title: plot title (optional, string)
  '''
  c = ['r','b','g']
  kwargs = dict(kde_kws={'linewidth':1})
  fig, ax = plt.subplots(figsize=[6*cm, 3*cm], dpi=300)

  for i, eGamma in enumerate(eGammas):
    vsamp, truth = vsamp_list[i], veffectList[i]
    if is_sample:
      sampEst = nanmean(nansum(vsamp, axis=2), axis=1)
    else:
      sampEst = vsamp
    # plot distribution
    sns.kdeplot(data=(sampEst-truth).T, label=r'$\tilde{\gamma}=$' + str(eGamma), ax=ax, linewidth=1, legend=False, color=c[i%len(c)])
  
  plt.title(title, fontdict={'fontsize':7}, pad=2)
  plt.xlabel(r'$\hat{v}-v_{\widetilde{\gamma}}}$', fontsize=5, labelpad=1)
  plt.ylabel('Density', fontsize=5, labelpad=1)
  ax.tick_params(axis='both', which='major', labelsize=5)
  plt.xlim([-15, 15])
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left', fontsize=5)
  plt.tight_layout(pad=0.25)
  plt.rcParams['figure.dpi'] = 300
  fig.subplots_adjust(right=0.7)
  plt.show()
