import pachinko_sim, bias_var, pstopSim
import numpy as np
import pylab as plt
from numpy.random import randint
from numpy.matlib import repmat
from numpy import cumsum, nanmean, nansum
import math

cm = 1/2.54  # centimeters in inches

nRows, nCols = 10, 9
nRews = 15
minRews, maxRews = 5, 24
rewSize = 1
addAbsorbingSt = True
extraTransOnSR = True

nExps = 100
nTrials = 50
nSamples = 1000
nRepeats = 50
maxSamples = 100
nTrajs = 10000
rho, beta = 1, 0
alpha = 0.01
lamb = 0.7
gamma = 0.9
pstop = 0.05
rewardModulateAlpha = 0.5
lrScheduleType = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.001}}

def runSim(nTrials, nSamples, nRows, nCols, nRews, maze, s, T, M, gamma, pstop=None,
            MFC=None, maxSamples=50, rho=None, beta=None, rewSize=1, 
            addAbsorbingSt=True, extraTransOnSR=True, verbose=False):
  if beta == 0:
    rewIdx = np.ravel_multi_index(np.where(maze == rewSize), (nRows, nCols))  # reward locations
    vtrue, _, vsamp, _, samped = bias_var.runSim(nTrials, nSamples, nRows, nCols, rewIdx.size, 
                                                  s, M, M,rewSize=rewSize, userandom=False, idx=rewIdx, MFC=MFC,
                                                  addAbsorbingSt=addAbsorbingSt, rho=rho, beta=beta, verbose=verbose)
  else:
    eGamma = pstop * gamma + (1-pstop)   # effective gamma
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
    samped, vsamp, _, vtrue = pstopSim.runSim(nRepeats, nTrials, nRows, nCols, nRews, s, M, Meffect, pstop, userandom=False, maze=maze, 
                                          MFC=MFC, maxSamples=maxSamples, rho=rho, beta=beta)
  return vtrue, vsamp, samped

def getRolloutSampledValues(vsamp, samped, beta):
  nRepeats, nTrials, nSamples = vsamp.shape
  mean = np.zeros((nRepeats, nTrials * nSamples))
  for e in range(nRepeats):
    s = 0
    while s < nTrials * nSamples:
      t = math.ceil((s+1) / nSamples)
      r = (s+1) % nSamples
      if samped[e,t-1,r-1] <= 0:   # the current trial ended, start inspecting the next
        if t < nTrials:
          t += 1
          r = 1
        else:
          break
      if r == 0: 
        r = nSamples
      if t == 1: 
        est = np.expand_dims(nansum(vsamp[e,t-1,:r])*beta, axis=0)
      else: 
        est = np.append(nansum(vsamp[e,:t-1,:], axis=1)*beta, np.expand_dims(nansum(vsamp[e,t-1,:r])*beta, axis=0), axis=0)
      mean[e,s] = nanmean(est)
      s += 1
    mean[e,s:] = mean[e,s-1]
  return mean

def getSampleBasedChoice(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=0):
  if beta == 0:   # i.i.d. samples
    nTrials, nSamples = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, nSamples, 1).T
    vtrue_s2 = repmat(vtrue_s2, nSamples, 1).T
    v1 = cumsum(vsamp_s1, axis=1)/np.arange(1, nSamples+1)
    v2 = cumsum(vsamp_s2, axis=1)/np.arange(1, nSamples+1)
  else:    # generalized rollout
    nRepeats, nTrials, nSamples = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, nTrials * nSamples, 1).T
    vtrue_s2 = repmat(vtrue_s2, nTrials * nSamples, 1).T
    v1 = getRolloutSampledValues(vsamp_s1, samped_s1, beta)
    v2 = getRolloutSampledValues(vsamp_s2, samped_s2, beta)
  correct = np.where(vtrue_s1 >= vtrue_s2, 1, 2)   # correct responses
  choice = np.where(v1 >= v2, 1, 2)          # sample-based responses
  return correct, choice

def getCorrectRate(correct, choice, beta=0):
  '''
  correct: correct responses, dim = nExps x nTrials x nSamples if beta == 0, else dim = nExps x (nTrials * maxSamples)
  choice: sample-based responses, dim = nExps x nTrials x nSamples if beta == 0, else dim = nExps x (nTrials * maxSamples)
  '''
  scorer = np.where(choice == correct, 1, 0)
  # compute average correct rate
  avg = np.mean(scorer, axis=(0,1))
  # compute worst correct rate
  trialAvg = np.mean(scorer, axis=1)
  worst = np.min(trialAvg, axis=0)
  worst_idx = np.argmin(trialAvg, axis=0)
  return avg, worst, worst_idx

def plotSampleBasedCorrectRate(avg, worst, rews):
  '''
  avg: average correct rate, len = nSamples
  worst: worst correct rate by maze, len = nSamples
  '''
  nr, nSamples = avg.shape
  fig, ax = plt.subplots(1, figsize=[5*cm, 4*cm], dpi=300)
  for i in range(nr):
    ax.semilogx(np.arange(0,nSamples), avg[i], linewidth=1, linestyle='-', label=str(rews[i]) + ' rewards')
  # ax.semilogx(np.arange(0,nSamples), worst, linewidth=1, color='black', linestyle='dashed', label='Worst case')

  ax.tick_params(labelsize=5, length=2, pad=0)
  ax.set_xlabel('Number of samples', fontdict={'fontsize':7}, labelpad=1)
  ax.set_ylabel('Sample-based correct rate', fontdict={'fontsize':7}, labelpad=2)
  ax.set_ylim(0,1)
  ax.legend(loc='lower right', fontsize=5)
  plt.tight_layout()
  plt.show()

def getSampleBasedReward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=0):
  if beta == 0:   # i.i.d. samples
    nTrials, nSamples = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, nSamples, 1).T
    vtrue_s2 = repmat(vtrue_s2, nSamples, 1).T
    v1 = cumsum(vsamp_s1, axis=1)/np.arange(1, nSamples+1)
    v2 = cumsum(vsamp_s2, axis=1)/np.arange(1, nSamples+1)
  else:    # generalized rollout
    nRepeats, nTrials, nSamples = vsamp_s1.shape
    vtrue_s1 = repmat(vtrue_s1, nTrials * nSamples, 1).T
    vtrue_s2 = repmat(vtrue_s2, nTrials * nSamples, 1).T
    v1 = getRolloutSampledValues(vsamp_s1, samped_s1, beta)
    v2 = getRolloutSampledValues(vsamp_s2, samped_s2, beta)
  values = np.transpose(np.array([vtrue_s1, vtrue_s2]), (1,2,0))    # expected option values
  choice = np.where(v1 >= v2, 0, 1)                                # sample-based choice
  exp_rew = np.zeros(choice.shape)
  for i in range(choice.shape[0]):
    for j in range(choice.shape[1]):
      exp_rew[i,j] = values[i,j,choice[i,j]]
  return values, choice, exp_rew

def getPerformance(values, exp_rew, beta=0):
  max_val = np.max(values, axis=-1)
  assert(max_val.shape == exp_rew.shape)
  return np.nanmean(exp_rew / max_val, axis=(0,1))

def plotSampleBasedPerformance(avg, rews):
  '''
  avg: average performance, len = nSamples
  '''
  nr, nSamples = avg.shape
  fig, ax = plt.subplots(1, figsize=[5*cm, 2.8*cm], dpi=300)
  for i in range(nr):
    ax.semilogx(np.arange(0,nSamples), avg[i], linewidth=1, linestyle='--', label=str(rews[i]) + ' rewards')
  # ax.semilogx(np.arange(0,nSamples), worst, linewidth=1, color='black', linestyle='dashed', label='Worst case')

  ax.tick_params(labelsize=5, length=2, pad=0)
  ax.set_xlabel('Number of samples', fontdict={'fontsize':5}, labelpad=2)
  ax.set_ylabel('% of max rewards', fontdict={'fontsize':5}, labelpad=2)
  ax.set_ylim(0,1)
  ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  # ax.legend(loc='lower right', fontsize=5)

  fig.subplots_adjust(right=0.6)
  plt.tight_layout(pad=0.25)
  plt.savefig('mod-dash.pdf', dpi=300)
  plt.show()

def main():
  if beta == 0: correct, choice = np.zeros((nExps, nTrials, nSamples, 2)), np.zeros((nExps, nTrials, nSamples))
  else: correct, choice = np.zeros((nExps, nRepeats, nTrials * maxSamples, 2)), np.zeros((nExps, nRepeats, nTrials * maxSamples))
  # pick two states
  s1, s2 = (0,4), (0,5)
  # compute true transition matrix and SR
  nStates = nRows * nCols
  T = pachinko_sim.initTrans(nStates, nRows, nCols, addAbsorbingSt=addAbsorbingSt)
  M, M_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
  MTF_id = np.identity(M.shape[0])
  rews = [1, 5, 10, 20]
  avg = np.zeros((len(rews), choice.shape[-1]))
  for i, nr in enumerate(rews):
    for e in range(nExps):
      if nExps >= 10 and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
      # generate each maze
      if gamma == 0 and (beta == 0 or pstop == 1):
        maze = pachinko_sim.initMaze(nRows, nCols, nr, reachable=[s1,s2], rewSize=rewSize, userandom=True, forceRow=1)
      else:
        maze = pachinko_sim.initMaze(nRows, nCols, nr, reachable=[s1,s2], rewSize=rewSize, userandom=True)
      # draw samples from s1                             
      vtrue_s1, vsamp_s1, samped_s1 = runSim(nTrials, nSamples, nRows, nCols, nRews, maze, s1, T, M, gamma,
                                  pstop=pstop, MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta, rewSize=rewSize)
      # draw samples from s2
      vtrue_s2, vsamp_s2, samped_s2 = runSim(nTrials, nSamples, nRows, nCols, nRews, maze, s2, T, M, gamma,
                                  pstop=pstop, MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta, rewSize=rewSize)
      # get correct responses and sample-based choices
      correct[e,:], _, choice[e,:] = getSampleBasedReward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=beta)
    # compute overall correctness rate (average and worst case)
    avg[i,:] = getPerformance(correct, choice, beta=beta)
  # plot graph
  plotSampleBasedPerformance(avg, rews)

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

def sampleValueEst(maze, nSamples, gamma, s, M, MTD, alphaMod):
  nRows, nCols = maze.shape
  MTF_id = np.identity(M.shape[0])
  rewIdx = np.ravel_multi_index(np.where(maze > 0), (nRows, nCols))
  _, vtrue, vsamp, _, samped = bias_var.runSim2(nRepeats, nSamples, nRows, nCols, rewIdx.size, maze,
                                                    s, MTD, M, MFC=MTF_id, addAbsorbingSt=addAbsorbingSt, 
                                                    rho=rho, beta=beta, verbose=False, mod=alphaMod, modFrac=0)
  return vtrue, vsamp, samped

def learnSR(maze, s, scheduler):
  nStates = maze.size + addAbsorbingSt
  for t in range(nTrajs):
    # generate trajectories
    traj = pachinko_sim.getPachinkoRandomTraj(maze, s, addAbsorbingSt=addAbsorbingSt)
    # obtain intermediate SR
    if t == 0: MTD = np.identity(nStates)
    MTD, lr, alphaMod = updateSR(t, alpha, rewardModulateAlpha, scheduler, maze, traj, 
                                  MTD, gamma, lamb, addAbsorbingSt, extraTransOnSR)
  return MTD

def sampIntermediateSR(maze, s, M, scheduler):
  if nTrajs < 5: 
    vtrue, vsamp, samped = np.zeros(nTrials), np.zeros((nTrials, nSamples)), np.zeros((nTrials, nSamples))
    for j in range(nTrials):
      MTD = learnSR(maze, s, scheduler)
      vtrue[j], vsamp[j,:], samped[j,:] = sampleValueEst(maze, nSamples, gamma, s, M, MTD, rewardModulateAlpha)
  else: 
    MTD = learnSR(maze, s, scheduler)
    # print(maze)
    # pachinko_sim.plotSR(maze, M, s, np.arange(0, nRows * nCols).reshape(maze.shape), addAbsorbingSt=addAbsorbingSt)
    # pachinko_sim.plotSR(maze, MTD, s, np.arange(0, nRows * nCols).reshape(maze.shape), addAbsorbingSt=addAbsorbingSt)
    vtrue, vsamp, samped = sampleValueEst(maze, nSamples, gamma, s, M, MTD, rewardModulateAlpha)
  return vtrue, vsamp, samped

def main2():
  if nTrajs < 5: correct, choice = np.zeros((nExps, nTrials, nSamples, 2)), np.zeros((nExps, nTrials, nSamples))
  else: correct, choice = np.zeros((nExps, nRepeats, nSamples, 2)), np.zeros((nExps, nRepeats, nSamples))
  # pick two states
  s1, s2 = (0,4), (0,5)
  # compute true transition matrix and SR
  nStates = nRows * nCols
  T = pachinko_sim.initTrans(nStates, nRows, nCols, addAbsorbingSt=addAbsorbingSt)
  M, M_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
  scheduler = schedulers[lrScheduleType]
  rews = [1, 5, 10, 20]
  avg = np.zeros((len(rews), choice.shape[-1]))
  for i, nr in enumerate(rews):
    for e in range(nExps):
      if nExps >= 10 and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
      # generate each maze
      if gamma == 0 and (beta == 0 or pstop == 1):
        maze = pachinko_sim.initMaze(nRows, nCols, nr, reachable=[s1,s2], rewSize=rewSize, userandom=True, forceRow=1)
      else:
        maze = pachinko_sim.initMaze(nRows, nCols, nr, reachable=[s1,s2], rewSize=rewSize, userandom=True)
      # draw samples from s1                             
      vtrue_s1, vsamp_s1, samped_s1 = sampIntermediateSR(maze, s1, M, scheduler)
      # draw samples from s2
      vtrue_s2, vsamp_s2, samped_s2 = sampIntermediateSR(maze, s2, M, scheduler)
      # get correct responses and sample-based choices
      correct[e,:], _, choice[e,:] = getSampleBasedReward(vtrue_s1, vtrue_s2, vsamp_s1, vsamp_s2, samped_s1, samped_s2, beta=beta)
    # compute overall correctness rate (average and worst case)
    avg[i,:] = getPerformance(correct, choice, beta=beta)
  # plot graph
  plotSampleBasedPerformance(avg, rews)

main2()
