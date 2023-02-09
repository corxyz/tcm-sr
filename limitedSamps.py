import bias_var, pachinko_sim
import numpy as np
import pylab as plt
from scipy.stats import sem
from numpy.matlib import repmat
from numpy.random import randint
from matplotlib.gridspec import GridSpec
from numpy import percentile, nanmean

import math

cm = 1/2.54  # centimeters in inches

nRows, nCols = 10, 9
minRews, maxRews = 5, 24
rewSize = 1
rewIdx0 = [20,24,30,32,40]
# rewIdx0 = [38,40,42]
s0 = (0,4)
addAbsorbingSt = True
extraTransOnSR = True

nExps = 10                   # number of mazes
nSamples = 1000               # number of samples from each SR
rho, beta = 1, 0
alpha = 1
lamb = 0.7
gammas = [0.9, 0.9]
rewardModulateAlphas = [None, 2]
lrScheduleType = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.5}}

nTrials = 1              # number of trajectories for each maze
nTrajs = 100
trialsToPlot = [1,2,3,4]
stateIdx = np.arange(0, nRows * nCols).reshape((nRows, nCols))
nStates = nRows * nCols + addAbsorbingSt

def updateSR(k, alpha, alphaMod, scheduler, maze, traj, Minit, gamma, lamb, addAbsorbingSt, extraTransOnSR):
  if lrScheduleType == 'step':
    if (k+1) % scheduler['epochs'] == 0: # apply step decay
      alpha *= scheduler['rate']
      if alphaMod: 
        alphaMod *= scheduler['rate']
  elif lrScheduleType == 'exp':
    alpha *= np.exp(-scheduler['rate'] * k)
    if alphaMod: 
      alphaMod *= np.exp(-scheduler['rate'] * k)
  MTD = bias_var.getTDSR(maze, traj, Minit, rewSize=rewSize, alpha=alpha, gamma=gamma, lamb=lamb, 
                          rewardThreshold=0, rewardModulateAlpha=alphaMod,
                          addAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
  return MTD, alpha, alphaMod

def sampleValueEst(maze, nSamples, gamma, M, MTD, alphaMod):
  nRows, nCols = maze.shape
  MTF_id = np.identity(M.shape[0])
  rewIdx = np.ravel_multi_index(np.where(maze > 0), (nRows, nCols))
  _, vtrue, vsamp, (rCnt, nrCnt), _ = bias_var.runSim2(1, nSamples, nRows, nCols, rewIdx.size, maze,
                                                    s0, MTD, M, MFC=MTF_id, addAbsorbingSt=addAbsorbingSt, 
                                                    rho=rho, beta=beta, verbose=False, mod=alphaMod, modFrac=0)
  return vtrue, vsamp

def getValueBiasPctl(vsamp, vtrue):
  # vsamp dimensions: nExps x nTrajs x nSamples
  _, _, nSamples = vsamp.shape
  percentiles = [2.5, 25, 50, 75, 97.5]
  bias = vsamp - np.repeat(vtrue[:,:,np.newaxis], nSamples, axis=2)
  mean = np.zeros(nSamples)
  pctl = np.zeros((len(percentiles), nSamples))
  var = np.zeros(nSamples)
  for s in range(nSamples):
    mean[s] = nanmean(nanmean(bias[:,:,:s+1],axis=2))
    pctl[:,s] = percentile(nanmean(bias[:,:,:s+1],axis=2), percentiles)
    var[s] = np.var(nanmean(bias[:,:,:s+1],axis=2))
  return mean, pctl, var, bias

def plotValueEstLimitedSamps(nSamples, vtrues, vsamps, beta, gammas=None, rewardModulateAlphas=None, xabs_max=2):
  nGammas, nExps, nTrajs, nTrials = vtrues.shape
  nRows, nCols = 2*nGammas, nTrials
  fig = plt.figure(1, figsize=[nCols*3*1.3*cm, nRows*1.5*cm], dpi=300)
  gs = GridSpec(nRows, nCols*3, figure=fig, hspace=1.25, wspace=0.75)
  plt.clf()

  axes, axes2 = [], []
  ymax, ymax2 = 0, 6
  for i, gamma in enumerate(gammas):
    for k in range(nTrials):
      vsamp, vtrue = vsamps[i,:,:,k,:], vtrues[i,:,:,k]
      mean, pctl, var, bias = getValueBiasPctl(vsamp, vtrue)
      for j, nsamp in enumerate([10, 100, 1000]):
        if nSamples >= nsamp:
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
      modAlpha = None
      if rewardModulateAlphas: modAlpha = rewardModulateAlphas[i]
      if modAlpha: 
        ax.set_title(r'$\alpha_{mod}=$' + str(modAlpha), size=5, pad=0)
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
      ax.set_xlabel(r'$n$', fontdict={'fontsize':5}, labelpad=0)
      if k == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
      else: ax.tick_params(labelleft=False)
      ymax2 = max(max(ax.get_ylim()), ymax2)
      # ax.set_ylim([mean[-1]-5,mean[-1]+5])
    ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
  
  ymax2 = math.floor(ymax2)
  for ax in axes2:
    ax.set_ylim([-ymax2, ymax2])
  
  fig.subplots_adjust(right=0.8, top=0.92, bottom=0.12)
  # plt.savefig('limited_samps.pdf', dpi=300)
  # plt.tight_layout(pad=0.5)
  plt.show()

def plotSR(maze, traj, MTD):
  nRows, nCols = maze.shape
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[5*cm, 5*cm], dpi=300)
  # plot the board with traj and rewards marked
  for s in traj[:-addAbsorbingSt]:
    ax0.plot(s[1]+0.5, nRows-s[0]-0.5, 'kx', markersize=4, mew=0.5)
  rewR, rewC = np.where(maze > 0)
  for j, r in enumerate(rewR):
    c = rewC[j]
    ax0.plot(c+0.5, nRows-r-0.5, 'm*', markersize=4, mew=0.5)
  ax0.set_xticks(np.arange(0, nCols+1, 1))
  ax0.set_yticks(np.arange(0, nRows, 1))
  ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax0.grid()
  ax0.set_aspect('equal', adjustable='box')
  # plot the learned SR
  sr = MTD[4,:-addAbsorbingSt].reshape(maze.shape)
  # constrain all values to the prespecified range for plotting
  im = ax1.imshow(sr, vmin=0, vmax=.075, cmap="Greys")
  ax1.set_xticks(np.arange(-.5, nCols, 1))
  ax1.set_yticks(np.arange(-.5, nRows, 1))
  ax1.set_xticklabels(np.arange(0, nCols+1, 1))
  ax1.set_yticklabels(np.arange(0, nRows+1, 1))
  ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax1.grid()
  ax1.set_aspect('equal', adjustable='box')

  plt.tight_layout()
  plt.show()

def main():
  MTDs = np.zeros((len(gammas), nExps, nTrajs, nTrials, nStates, nStates))
  vsamps = np.zeros((len(gammas), nExps, nTrajs, nTrials, nSamples))
  vtrues = np.zeros((len(gammas), nExps, nTrajs, nTrials))
  scheduler = schedulers[lrScheduleType]
  for i, gamma in enumerate(gammas):
    # compute true transition matrix and SR
    T = pachinko_sim.initTrans(nStates, nRows, nCols, addAbsorbingSt=addAbsorbingSt)
    M, _, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
    for e in range(nExps):
      # generate each maze
      maze = pachinko_sim.initMaze(nRows, nCols, randint(minRews, maxRews), 
                                    rewSize=rewSize, userandom=True, reachable=s0, nRowExcl=2)
      for j in range(nTrajs):
        # generate trajectories
        for t in range(nTrials):
          traj = pachinko_sim.getPachinkoRandomTraj(maze, s0, addAbsorbingSt=addAbsorbingSt)
          # obtain intermediate SR
          if t == 0: 
            MTD = np.identity(nStates)
            lr = alpha
            alphaMod = rewardModulateAlphas[i]
          else: 
            MTD = MTDs[i,e,j,t-1,:]
          MTDs[i,e,j,t,:], lr, alphaMod = updateSR(t, lr, alphaMod, scheduler, maze, traj, MTD, gamma, lamb, addAbsorbingSt, extraTransOnSR)
          # plotSR(maze, traj, MTDs[i,e,j,t,:])
          # plotSR(maze, traj, M)
          # estimate value via sampling
          vtrue, vsamp = sampleValueEst(maze, nSamples, gamma, M, MTDs[i,e,j,t,:], rewardModulateAlphas[i])
          # record
          vtrues[i,e,j,t] = vtrue
          vsamps[i,e,j,t,:] = vsamp
  # plot value estimate convergence
  plotValueEstLimitedSamps(nSamples, vtrues, vsamps, beta, 
                            gammas=gammas, rewardModulateAlphas=rewardModulateAlphas, xabs_max=5)

main()

