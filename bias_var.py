import numpy as np
from numpy.random import rand, randn, random
from numpy.matlib import repmat
from numpy import percentile, nanmean, std, nansum, nanpercentile
import pylab as plt
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

import math, itertools
import pachinko_sim

cm = 1/2.54  # centimeters in inches

def initMaze(nRows, nCols, nRews):
  return pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)

def initTrans(nRows, nCols, addAbsorbingSt=False, reversible=False):
  nStates = nRows * nCols
  if addAbsorbingSt: 
    T = np.zeros([nStates+1, nStates+1])
    T[nStates, nStates] = 1
  else: T = np.zeros([nStates, nStates])

  for s in range(nStates):
    i, j = np.unravel_index(s, (nRows, nCols))
    neighbors = []
    if i < nRows-1:   # not at the bottom
      if j > 0 and j < nCols-1: # not at the walls
        neighbors += [(i+1, j-1), (i+1, j+1)]
        if reversible and i > 0: neighbors += [(i-1, j-1), (i-1, j+1)]
      elif j == 0:                # left wall
        neighbors.append((i+1, j+1))
        if reversible and i > 0: neighbors.append((i-1, j+1))
      else:                       # right wall
        neighbors.append((i+1, j-1))
        if reversible and i > 0: neighbors.append((i-1, j-1))
      for neighbor in neighbors:
        T[s, np.ravel_multi_index(neighbor, (nRows, nCols))] = 1/len(neighbors)
    else:               # at bottom
      if addAbsorbingSt: T[s, nStates] = 1
      else: T[s, s] = 1
  return T

def getSR(T, gamma=1, hasAbsorbingSt=False, extraTransOnSR=False):
  M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  M_gamma1 = np.linalg.inv(np.identity(T.shape[0]) - 0.9999999999*T)
  if extraTransOnSR:
    M = np.matmul(M, T)
    M_gamma1 = np.matmul(M_gamma1, T)
  return M, M_gamma1, T

def getTDSR(maze, traj, Minit, rewSize=1, alpha=0.1, gamma=1, lamb=0, addAbsorbingSt=True,
            extraTransOnSR=True, rewardThreshold=None, rewardModulateAlpha=None):
  numRows, numCols = maze.shape
  nStates = maze.size
  M = Minit.copy()                  # SR
  e = np.zeros(nStates+addAbsorbingSt)             # eligibility trace
  for n, curPos in enumerate(traj[:-addAbsorbingSt]):
    curIdx = np.ravel_multi_index(curPos,(numRows,numCols))
    nextPos = traj[n+extraTransOnSR]
    if nextPos[0] == numRows:  # absorbing state
      nextIdx = maze.size
    else:
      nextIdx = np.ravel_multi_index(nextPos,(numRows,numCols))
    # modulate learning rate if necessary
    if rewardModulateAlpha and nextIdx < maze.size and maze[nextPos] > rewardThreshold:
      lr = rewardModulateAlpha
    else:
      lr = alpha
    s, s1 = np.zeros(nStates+addAbsorbingSt), np.zeros(nStates+addAbsorbingSt)
    s[curIdx] = 1
    s1[nextIdx] = 1
    sT, s1T = np.expand_dims(s, axis=1).T, np.expand_dims(s1, axis=1).T
    e = gamma * lamb * e + s
    M = M + lr * np.matmul(np.expand_dims(e, axis=1), (s1T + gamma * np.matmul(s1T, M) - np.matmul(sT, M)))
  return M

def getHebbianSR(maze, traj, Minit, rewSize=1, alpha=0.1, gamma=1, lamb=0, addAbsorbingSt=True,
                  extraTransOnSR=True, rewardThreshold=None, rewardModulateAlpha=None):
  numRows, numCols = maze.shape
  nStates = maze.size
  M = Minit.copy()                  # SR
  e = np.zeros(nStates+addAbsorbingSt)             # eligibility trace
  for n, curPos in enumerate(traj[:-addAbsorbingSt]):
    curIdx = np.ravel_multi_index(curPos,(numRows,numCols))
    nextPos = traj[n+extraTransOnSR]
    if nextPos[0] == numRows:  # absorbing state
      nextIdx = maze.size
    else:
      nextIdx = np.ravel_multi_index(nextPos,(numRows,numCols))
    # modulate learning rate if necessary
    if rewardModulateAlpha and nextIdx < maze.size and maze[nextPos] > rewardThreshold:
      lr = rewardModulateAlpha
    else:
      lr = alpha
    s, s1 = np.zeros(nStates+addAbsorbingSt), np.zeros(nStates+addAbsorbingSt)
    s[curIdx] = 1
    s1[nextIdx] = 1
    sT, s1T = np.expand_dims(s, axis=1).T, np.expand_dims(s1, axis=1).T
    e = gamma * lamb * e + s
    M = M + lr * np.matmul(np.expand_dims(e, axis=1), s1T) - lr * np.matmul(np.matmul(np.expand_dims(s, axis=1), sT), M)
  return M

def runSim(nExps, nSamples, nRows, nCols, nRews, s0, M, M_gamma1,
            rewSize=1, userandom=True, idx=[2,8,14,27,34,49,52,65,79,87], randomRewards=False, reachable=None, 
            MFC=None, addAbsorbingSt=True,
            rho=None, beta=None, ensureCnormIs1=True, verbose=True):
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(nExps)
  vgamma1 = np.zeros(nExps)
  vsamp = np.zeros((nExps,nSamples))
  sampRow = np.zeros((nExps,nSamples), dtype=int)
  sampCol = np.zeros((nExps,nSamples), dtype=int)
  rCnt, nrCnt = np.zeros((nExps,3)), np.zeros((nExps,3))

  for e in range(nExps):
    if verbose and nExps >= 10 and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, rewSize=rewSize, userandom=userandom, idx=idx, reachable=reachable)
    rvec = maze.flatten() # 1-step rewards
    if randomRewards: rvec = 1 + randn(rvec.size)
    if addAbsorbingSt: rvec = np.append(rvec, 0)

    stateIdx = np.arange(0, nRows * nCols).reshape(maze.shape)    # state indices
    stim = np.identity(nRows * nCols + addAbsorbingSt)

    vtrue[e] = np.dot(M[stateIdx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    vgamma1[e] = np.dot(M_gamma1[stateIdx[s0],:], rvec)       # Compute the actual value function (as v=Mr)

    c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)

    sampleIdx = np.negative(np.ones(nSamples, dtype=int))
    for i in range(nSamples):
      # define sampling distribution
      a = np.matmul(M.T,c)[:-addAbsorbingSt]
      P = a / np.sum(a)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(P))[0]
      sampleIdx[i] = tmp[0]
      sampRow[e,i], sampCol[e,i] = np.unravel_index(sampleIdx[i], (nRows, nCols))
      f = stim[:,sampleIdx[i]]

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if ensureCnormIs1:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[sampleIdx[sampleIdx >= 0]]
    # count samples drawn from rewarding vs non-rewarding states
    rCnt[e,0], nrCnt[e,0] = sum(rewsamp[:10] > 0), sum(rewsamp[:10] == 0)
    rCnt[e,1], nrCnt[e,1] = sum(rewsamp[:100] > 0), sum(rewsamp[:100] == 0)
    rCnt[e,2], nrCnt[e,2] = sum(rewsamp[:1000] > 0), sum(rewsamp[:1000] == 0)
    rewsamp = np.append(rewsamp, np.zeros(nSamples - len(rewsamp)))
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[stateIdx[s0],:]) 

  return vtrue, vgamma1, vsamp, (rCnt, nrCnt), sampRow

def runSim2(nExps, nSamples, nRows, nCols, nRews, maze, s0, M, M_gamma1,
            # rewSize=1, userandom=True, idx=[2,8,14,27,34,49,52,65,79,87], randomRewards=False, reachable=None, 
            MFC=None, addAbsorbingSt=True,
            rho=None, beta=None, ensureCnormIs1=True, verbose=True, mod=False, modFrac=0):
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(nExps)
  vgamma1 = np.zeros(nExps)
  vsamp = np.zeros((nExps,nSamples))
  sampRow = np.zeros((nExps,nSamples), dtype=int)
  sampCol = np.zeros((nExps,nSamples), dtype=int)
  rCnt, nrCnt = np.zeros((nExps,3)), np.zeros((nExps,3))

  idx = np.ravel_multi_index(np.where(maze > 0), (nRows, nCols))

  for e in range(nExps):
    if verbose and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
    rvec = maze.flatten() # 1-step rewards
    if addAbsorbingSt: rvec = np.append(rvec, 0)

    stateIdx = np.arange(0, nRows * nCols).reshape(maze.shape)    # state indices
    stim = np.identity(nRows * nCols + addAbsorbingSt)

    vtrue[e] = np.dot(M[stateIdx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    vgamma1[e] = np.dot(M_gamma1[stateIdx[s0],:], rvec)       # Compute the actual value function (as v=Mr)

    c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)

    sampleIdx = np.negative(np.ones(nSamples, dtype=int))
    ps, qs = np.zeros(nSamples), np.zeros(nSamples)

    for i in range(nSamples):
      # define sampling distribution
      a = np.matmul(M.T,c)[:-addAbsorbingSt]
      Q = a / np.sum(a)
      assert np.abs(np.sum(Q)-1) < 1e-10, 'Q is not a valid probability distribution'

      # get reference distribution
      aref = np.matmul(M_gamma1.T,c)[:-addAbsorbingSt]
      P = aref / np.sum(aref)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(Q))[0]
      sampleIdx[i] = tmp[0]

      if sampleIdx[i] >= nRows * nCols: # sampled absorbing state:
        break
      else:
        sampRow[e,i], sampCol[e,i] = np.unravel_index(sampleIdx[i], (nRows, nCols))
        f = stim[:,sampleIdx[i]]
      
      if mod:
        if (random() < modFrac):
          sampleIdx[i] = idx[0]
          f = stim[:,sampleIdx[i]]
      
      # store sampling probabilities
      ps[i], qs[i] = P[sampleIdx[i]], Q[sampleIdx[i]] * (1-modFrac) if sampleIdx[i] != idx[0] else Q[sampleIdx[i]] * (1-modFrac) + modFrac

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if ensureCnormIs1:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[sampleIdx[sampleIdx >= 0]]
    rewsamp = np.append(rewsamp, np.zeros(nSamples - len(rewsamp)))
    # reweigh samples
    if mod: 
      rewsamp = np.divide(np.multiply(rewsamp, ps), qs)
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[stateIdx[s0],:])
    # count samples drawn from rewarding vs non-rewarding states
    rCnt[e,0], nrCnt[e,0] = sum(rewsamp[:10] > 0), sum(rewsamp[:10] == 0)
    rCnt[e,1], nrCnt[e,1] = sum(rewsamp[:100] > 0), sum(rewsamp[:100] == 0)
    rCnt[e,2], nrCnt[e,2] = sum(rewsamp[:1000] > 0), sum(rewsamp[:1000] == 0)

  return vtrue, vgamma1, vsamp, (rCnt, nrCnt), sampRow

def getValueEstPctl(vsamp, nSamples, beta):
  percentiles = [2.5, 25, 50, 75, 97.5]
  mean = np.empty(nSamples)
  pctl = np.empty((len(percentiles), nSamples))
  if beta == 0:
    for s in range(nSamples):
      mean[s] = nanmean(nanmean(vsamp[:,:s+1],axis=1))
      pctl[:,s] = percentile(nanmean(vsamp[:,:s+1],axis=1), percentiles)
  else:
    for s in range(nSamples):
      mean[s] = nanmean(nansum(vsamp[:,:s+1],axis=1))
      pctl[:,s] = percentile(nansum(vsamp[:,:s+1],axis=1), percentiles)
  return mean, pctl

def getBiasVarPctl(vtrue, vgamma1, vsamp, nSamples, s0, M, nRows, stateIdx):
  # compute bias
  vsamp_minus_vreal = vsamp - repmat(vtrue, nSamples, 1).T # express vsamp as a distance to vtrue
  vsamp_minus_vgamma1 = (vsamp / np.sum(M[stateIdx[s0],:])) * (nRows-1) - repmat(vgamma1, nSamples, 1).T # express vsamp as a distance to vgamma1

  # compute mean and percentiles of the estimator
  vsampBiasVar = dict()
  vsampBiasVar['vsamp_minus_vreal'] = vsamp_minus_vreal
  vsampBiasVar['vsamp_minus_vgamma1'] = vsamp_minus_vgamma1
  vsampBiasVar['vsamp_minus_vreal_prctiles'] = np.empty((5, nSamples))
  vsampBiasVar['vsamp_minus_vreal_mean']= np.empty(nSamples)
  vsampBiasVar['vsamp_minus_vreal_std'] = np.empty(nSamples)
  vsampBiasVar['vsamp_minus_vgamma1_prctiles'] = np.empty((5, nSamples))
  vsampBiasVar['vsamp_minus_vgamma1_mean'] = np.empty(nSamples)
  vsampBiasVar['vsamp_minus_vgamma1_std'] = np.empty(nSamples)
  percentiles = [2.5, 25, 50, 75, 97.5]
  for s in range(nSamples):
    vsampBiasVar['vsamp_minus_vreal_prctiles'][:,s] = percentile(nanmean(vsamp_minus_vreal[:,:s+1],axis=1), percentiles)
    vsampBiasVar['vsamp_minus_vreal_mean'][s] = nanmean(nanmean(vsamp_minus_vreal[:,:s+1],axis=1))
    vsampBiasVar['vsamp_minus_vgamma1_prctiles'][:,s] = percentile(nanmean(vsamp_minus_vgamma1[:,:s+1],axis=1), percentiles)
    vsampBiasVar['vsamp_minus_vgamma1_mean'][s] = nanmean(nanmean(vsamp_minus_vgamma1[:,:s+1],axis=1))
    vsampBiasVar['vsamp_minus_vreal_std'][s] = std(nanmean(vsamp_minus_vreal[:,:s+1],axis=1))
    vsampBiasVar['vsamp_minus_vgamma1_std'][s] = std(nanmean(vsamp_minus_vgamma1[:,:s+1],axis=1))
  
  return vsampBiasVar

def getValueEstPctl(vsamp, vtrue, beta):
  nExps, nSamples = vsamp.shape
  percentiles = [2.5, 25, 50, 75, 97.5]
  if beta == 0:
    bias = vsamp - repmat(vtrue, nSamples, 1).T
    mean = np.empty(nSamples)
    pctl = np.empty((len(percentiles), nSamples))
    for s in range(nSamples):
      mean[s] = nanmean(nanmean(bias[:,:s+1],axis=1))
      pctl[:,s] = nanpercentile(nanmean(bias[:,:s+1],axis=1), percentiles)
  else:
    mean = np.empty(nTrials*nSamples)
    pctl = np.empty((len(percentiles), nTrials*nSamples))
    bias = np.empty((nExps, nTrials*nSamples))
    for s in range(nTrials*nSamples):
      t = math.ceil((s+1) / nSamples) # compute number of trials included
      r = (s+1) % nSamples            # compute number of samples included that belong to the last (possibly incomplete) trial
      if r == 0: r = nSamples
      if t == 1: est = np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1)
      else: est = np.append(nansum(vsamp[:,:t-1,:], axis=2)*beta, np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1), axis=1)
      bias[:,s] = nanmean(est, axis=1) - vtrue
      mean[s] = nanmean(bias[:,s])
      pctl[:,s] = percentile(bias[:,s], percentiles)
  return mean, pctl, bias

def plotValueEst(nSamples, vtrues, vsamps, beta, gammas=None, rewardModulateAlphas=None, xabs_max=5):
  nRows, nCols = 2 * len(gammas), 1
  fig = plt.figure(1, figsize=[nCols*3*1.5*cm, nRows*1.5*cm], dpi=300)
  plt.clf()
  gs = GridSpec(nRows, nCols*3, figure=fig, hspace=1.1)

  axes, axes2 = [], []
  ymax, ymax2 = 0, 0
  for i, gamma in enumerate(gammas):
    vsamp, vtrue = vsamps[i], vtrues[i]
    mean, pctl, bias = getValueEstPctl(vsamp, vtrue, nSamples)
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
    modAlpha = None
    if rewardModulateAlphas: modAlpha = rewardModulateAlphas[i]
    if modAlpha: 
      ax.set_title(r'$\alpha_{mod}=$' + str(modAlpha), size=7, pad=5)
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

def plotValueEstLearnedSR(nSamples, vtrues, vsamps, beta, gammas=None, rewardModulateAlphas=None, xabs_max=5):
  nRows, nCols = 2, len(gammas)
  fig = plt.figure(1, figsize=[nCols*3*2*cm, nRows*2.4*cm], dpi=300)
  gs = GridSpec(nRows, nCols*3, figure=fig, hspace=0.8)
  plt.clf()

  axes, axes2 = [], []
  ymax, ymax2 = 0, 6
  for i, gamma in enumerate(gammas):
    vsamp, vtrue = vsamps[i], vtrues[i]
    mean, pctl, bias = getValueEstPctl(vsamp, vtrue, beta)
    for j, nsamp in enumerate([10, 100, 1000]):
      if nSamples >= nsamp:
        ax = fig.add_subplot(gs[0,i*3+j])
        axes.append(ax)
        ax.set_title(r'$N=$' + str(nsamp), size=5, pad=2)
        # if nsamp < 100: tmp = np.concatenate([nanmean(bias[:,k:k+nsamp], axis=1) for k in range(0, nSamples-nsamp+1, nsamp)])
        # else: 
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
    modAlpha = None
    if rewardModulateAlphas: modAlpha = rewardModulateAlphas[i]
    if modAlpha: 
      ax.set_title(r'$\alpha_{mod}=$' + str(modAlpha), size=7, pad=2)
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
    if i == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
    else: ax.tick_params(labelleft=False)
    # ymax2 = max(max(ax.get_ylim()), ymax2)
    # ax.set_ylim([mean[-1]-5,mean[-1]+5])
  ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
    ax.set_xticks([-xabs_max,0,xabs_max])
  
  # ymax2 = math.floor(ymax2)
  for ax in axes2:
    ax.set_ylim([-xabs_max, xabs_max])
    ax.set_yticks([-xabs_max,0,xabs_max])
  
  fig.subplots_adjust(right=0.8, top=0.92, bottom=0.12)
  plt.savefig('f5c.pdf', dpi=300)
  # plt.tight_layout(pad=0.5)
  plt.show()

def plotMultipleBiasVar(nSamples, vtrues, vsamps, gammas=None):
  nRows, nCols = 1, len(gammas)
  plt.figure(1, figsize=[nCols*3, nRows*2])
  plt.clf()

  for i, gamma in enumerate(gammas):
    vsamp, vtrue = vsamps[i], vtrues[i]
    bias, pctl = getValueBiasPctl(vsamp, vtrue, nSamples)
    plt.subplot(nRows, nCols, i+1, title=r'$\gamma={}$'.format(gamma))
    lines1 = plt.semilogx(repmat(np.arange(0,nSamples),5,1).T, pctl.T)
    lines2 = plt.semilogx(np.arange(0,nSamples), bias)
    plt.grid(True)
    plt.setp(lines1[0], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='95% CI')
    plt.setp(lines1[4], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='50% CI')
    plt.setp(lines1[1], color=[0.3,0.3,0.3], linewidth=1, linestyle='--', label='Median')
    plt.setp(lines1[3], color=[0.3,0.3,0.3], linewidth=1, linestyle='--')
    plt.setp(lines1[2], color=[0.1,0.1,0.1], linewidth=1, linestyle='-', label='Mean')
    plt.setp(lines2, color=[0,0,0.8], linewidth=2, linestyle='-')
    plt.xlabel('# samples averaged')
    plt.ylabel(r'$\hat{v}-v_{true}$')
  
  plt.legend()
  plt.tight_layout()
  plt.show()

def plotBiasVar(nSamples, vsampBiasVar, gamma=1):
  vsamp_minus_vreal_prctiles = vsampBiasVar['vsamp_minus_vreal_prctiles']
  vsamp_minus_vreal_mean = vsampBiasVar['vsamp_minus_vreal_mean']
  vsamp_minus_vgamma1_prctiles = vsampBiasVar['vsamp_minus_vgamma1_prctiles']
  vsamp_minus_vgamma1_mean = vsampBiasVar['vsamp_minus_vgamma1_mean']
  vsamp_minus_vreal = vsampBiasVar['vsamp_minus_vreal']
  vsamp_minus_vgamma1 = vsampBiasVar['vsamp_minus_vgamma1']
  vsamp_minus_vreal_std = vsampBiasVar['vsamp_minus_vreal_std']
  vsamp_minus_vgamma1_std = vsampBiasVar['vsamp_minus_vgamma1_std']

  plt.figure(1, figsize=[15,10])
  plt.clf()

  plt.subplot(3,2,1,title=r'v ($\gamma$='+str(round(gamma,2))+')')
  lines1 = plt.semilogx(repmat(np.arange(0,nSamples),5,1).T, vsamp_minus_vreal_prctiles.T)
  lines2 = plt.semilogx(np.arange(0,nSamples), vsamp_minus_vreal_mean)
  plt.grid(True)
  plt.ylim([-3,3])
  plt.setp(lines1[0], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='95% CI')
  plt.setp(lines1[4], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='50% CI')
  plt.setp(lines1[1], color=[0.3,0.3,0.3], linewidth=1, linestyle='--', label='Median')
  plt.setp(lines1[3], color=[0.3,0.3,0.3], linewidth=1, linestyle='--')
  plt.setp(lines1[2], color=[0.1,0.1,0.1], linewidth=1, linestyle='-', label='Mean')
  plt.setp(lines2, color=[0,0,0.8], linewidth=2, linestyle='-')
  plt.xlabel('# samples averaged')
  plt.ylabel(r'$v_{est} - v_{true}$')
  plt.legend()

  plt.subplot(3,2,2,title=r'v ($\gamma$=1)')
  lines1 = plt.semilogx(repmat(np.arange(0,nSamples),5,1).T, vsamp_minus_vgamma1_prctiles.T)
  lines2 = plt.semilogx(np.arange(0,nSamples), vsamp_minus_vgamma1_mean)
  plt.grid(True)
  plt.ylim([-10,10])
  plt.setp(lines1[0], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='95% CI')
  plt.setp(lines1[4], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='50% CI')
  plt.setp(lines1[1], color=[0.3,0.3,0.3], linewidth=1, linestyle='--', label='Median')
  plt.setp(lines1[3], color=[0.3,0.3,0.3], linewidth=1, linestyle='--')
  plt.setp(lines1[2], color=[0.1,0.1,0.1], linewidth=1, linestyle='-', label='Mean')
  plt.setp(lines2, color=[0,0,0.8], linewidth=2, linestyle='-')
  plt.xlabel('# samples averaged')
  plt.ylabel(r'$v_{est} - v_{\gamma=1}$')
  plt.legend()

  plt.subplot(3,2,3)
  plt.hist(nanmean(vsamp_minus_vreal[:,:10],axis=1),bins=np.arange(-3,3,0.05),density=True,label='10 samples')
  if vsamp_minus_vreal.shape[0] >= 100:
    plt.hist(nanmean(vsamp_minus_vreal[:,:100],axis=1),bins=np.arange(-3,3,0.05),density=True,label='100 samples')
  if vsamp_minus_vreal.shape[0] >= 1000:
    plt.hist(nanmean(vsamp_minus_vreal[:,:1000],axis=1),bins=np.arange(-3,3,0.05),density=True,label='1000 samples')
  if vsamp_minus_vreal.shape[0] >= 10000:
    plt.hist(nanmean(vsamp_minus_vreal[:,:10000],axis=1),bins=np.arange(-3,3,0.05),density=True,label='10000 samples')
  plt.xlabel(r'Bias: $v_{est} - v_{true}$')
  plt.ylabel('Probability')
  plt.legend()

  plt.subplot(3,2,4)
  plt.hist(nanmean(vsamp_minus_vgamma1[:,:10],axis=1),bins=np.arange(-3,3,0.05),density=True,label='10 samples')
  if vsamp_minus_vgamma1.shape[0] >= 100:
    plt.hist(nanmean(vsamp_minus_vgamma1[:,:100],axis=1),bins=np.arange(-3,3,0.05),density=True,label='100 samples')
  if vsamp_minus_vgamma1.shape[0] >= 1000:
    plt.hist(nanmean(vsamp_minus_vgamma1[:,:1000],axis=1),bins=np.arange(-3,3,0.05),density=True,label='1000 samples')
  if vsamp_minus_vgamma1.shape[0] >= 10000:
    plt.hist(nanmean(vsamp_minus_vgamma1[:,:10000],axis=1),bins=np.arange(-3,3,0.05),density=True,label='10000 samples')
  plt.xlabel(r'Bias: $v_{est} - v_{true}$')
  plt.ylabel('Probability')
  plt.legend()

  plt.subplot(3,2,5)
  lines1 = plt.semilogx(np.arange(0,nSamples), vsamp_minus_vreal_std)
  lines2 = plt.semilogx(np.arange(0,nSamples), vsamp_minus_vgamma1_std)
  plt.setp(lines1, linewidth=2, label=r'$v_{est} - v_{true}$')
  plt.setp(lines2, linewidth=2, label=r'$v_{est} - v_{\gamma=1}$')
  plt.xlabel('# samples averaged')
  plt.ylabel('Dispersion of estimator (std)')
  plt.legend()

  plt.tight_layout()
  plt.show()

def plotSampledRows(sampRow, nRows, nExps, nSamples, beta, gamma, normalize=False, includeReference=True):
  # sampRow has dimension nExps x nSamples
  plt.figure(1, figsize=(12,3))
  plt.clf()
  x = np.arange(0,nRows)
  # avgCnt = np.zeros(nRows)
  # for r in x: avgCnt[r] = (sampRow.flatten() == r).sum()/float(nExps)
  # plt.bar(x, avgCnt)
  # plot histogram outlines for the first 5, 10, ..., nSamples samples
  for s in np.arange(5, nSamples+1, 5):
    d = sampRow[:, :s].flatten()
    if normalize: # plot density function with smoothing
      density = stats.gaussian_kde(d)
      plt.plot(x, density(x), label=r'$\beta=$' + str(beta) + ', ' + str(s) + ' samples')
    else:         # plot average counts across experiments
      avgCnt = np.zeros(nRows)
      for r in x:
        avgCnt[r] = (d == r).sum()/float(nExps)
      plt.plot(x, avgCnt, label=r'$\beta=$' + str(beta) + ', ' + str(s) + ' samples')

  # plot reference lines for beta = 0 and beta = 1
  if includeReference:
    beta0 = np.array([(1-gamma) * gamma ** i for i in range(nRows)])
    beta1 = np.array([1/nRows for i in range(nRows)])
    plt.plot(x, beta0, label=r'$\beta=0$' + ' (density)')
    plt.plot(x, beta1, label=r'$\beta=1$' + ' (density')

  plt.xticks(np.arange(1, nRows+1, max(nRows//10,1)))
  plt.xlabel('row number')
  if normalize: plt.ylabel('density')
  else: plt.ylabel('average samples per experiment')
  plt.title('sampling distribution (' + r'$\gamma=$' + str(gamma) + ')')
  plt.legend()
  plt.show()

def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

def plotRecency(nExps, nRows, nCols, nRews, s0, M, MFC=None, rho=None, beta=None, addAbsorbingSt=False):
  recalls = np.zeros(nExps)
  for e in range(nExps):
    stateIdx = np.arange(0, nRows * nCols).reshape((nRows, nCols))    # state indices
    stim = np.identity(nRows * nCols + addAbsorbingSt)
    c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)

    curRow, curCol = s0
    for i in range(nRows-s0[0]-1):
      # proceed by one step
      curRow, curCol = pachinko_sim.randomStepPachinko(nRows, nCols, curRow, curCol)
      f = stim[:,stateIdx[curRow, curCol]]
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
    recalls[e] = np.unravel_index(samp, (nRows, nCols))[0]

  plt.figure(1,figsize=[3.75*cm, 3.75*cm], dpi=300)
  plt.clf()
  p = np.zeros(nRows)
  for i in range(nRows):
    p[i] = (recalls == i).sum()/nExps
  plt.plot(np.arange(1,nRows+1), p, 'ko-', markersize=2.5, linewidth=0.7)
  plt.xticks(np.arange(0,nRows+1,2), fontsize=5)
  plt.yticks(np.linspace(0, 0.8, 5), fontsize=5)
  plt.xlabel('Row', fontsize=5)
  plt.ylabel('Probability', fontsize=5)
  ax = plt.gca()
  ax.tick_params(length=2, direction='in')
  make_square_axes(ax)
  plt.tight_layout()
  plt.show()

def plotSampleCRP(sampRow, nRows, maxDist=9, omitZero=True):
  nExps, nSamples = sampRow.shape
  x = np.arange(-maxDist, maxDist+1)  # samples could come from row 1 to nRows-1 and we restrict it to the range of maxDist
  allCount = np.zeros(nRows * 2 - 1)
  count = np.zeros((nExps, maxDist * 2 + 1))  # compute CRP curve for each experiment
  # accumulate counts of relative positions of consecutive samples
  for e in range(nExps):
    samps = sampRow[e,:]
    for i in range(nSamples):
      if samps[i] < 0: break    # bottom reached
      if i == 0:  # i.i.d. samples or the first sample
        relPos = samps[i] if samps[i] > 0 else maxDist + 1
      else:
        relPos = samps[i] - samps[i-1]
      if abs(relPos) < nRows: allCount[relPos + nRows - 1] += 1
      if omitZero and relPos == 0 or abs(relPos) > maxDist: continue
      count[e, relPos + maxDist] += 1

  # compute density, mean, and standard error across experiments
  sum_of_rows = count.sum(axis=1)
  y = count / sum_of_rows[:, np.newaxis]
  if omitZero: y[:, maxDist] = np.nan
  mean = np.nanmean(y, axis=0)
  se = stats.sem(y, nan_policy='omit')
  # plot graph
  plt.figure(1,figsize=[3.5*cm, 3.5*cm], dpi=300)
  plt.clf()
  # plt.subplot(2,1,1)
  # plt.bar(np.arange(-nRows+1, nRows), allCount)
  # plt.xticks(np.arange(-nRows+1, nRows, max(nRows//10,1)), fontsize=5)
  # plt.yticks([])
  # plt.xlabel('row', fontsize=5)
  # plt.ylabel('n', fontsize=5)
  # plt.subplot(2,1,2)
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
