import bias_var, pachinko_sim
import numpy as np
import pylab as plt
from scipy.stats import sem
from numpy.matlib import repmat
from numpy.random import randint

cm = 1/2.54  # centimeters in inches

nRows, nCols = 10, 9
minRews, maxRews = 20, 40
rewSize = 1
rewIdx0 = [20,24,30,32,40]
# rewIdx0 = [38,40,42]
s0 = (0,4)
addAbsorbingSt = True
extraTransOnSR = True

nExps = 100
# nRepeats = 1
nSamples = 1000
rho, beta = 1, 0
alpha = 0.01
lamb = 0.7
gammas = [0.9, 0.9]
rewardModulateAlphas = [None, 0.5]
lrScheduleType = 'exp'       # implemented: step, exp
schedulers = {'step': {'epochs': 10, 'rate': 0.9},
              'exp':  {'rate': 0.001}}

nTrials = 10
trialsToPlot = [1,2,3,4,nTrials]
mazes = [None] * (nExps+1)
trajs = [[None] * nTrials for e in range(nExps+1)]
stateIdx = np.arange(0, nRows * nCols).reshape((nRows, nCols))

trueT = bias_var.initTrans(nRows, nCols, addAbsorbingSt=addAbsorbingSt)
nStates = nRows * nCols + addAbsorbingSt
Ts = np.zeros((len(gammas), nExps+1, len(trialsToPlot), nStates, nStates))

trueM_idx = 0  # index (w.r.t. the trialsToPlot array above) of iteratively learned SR to be used in evaluation 

def generateMazeTrajs():
  # generate mazes and sample trajectories
  print('Generating training data...', end='')
  for e in range(nExps+1):
    if nExps >= 10 and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%', end=' ')
    if e == 0:  # demo maze
      mazes[e] = pachinko_sim.initMaze(nRows, nCols, len(rewIdx0), rewSize=rewSize, userandom=False, idx=rewIdx0)
    else:
      mazes[e] = pachinko_sim.initMaze(nRows, nCols, randint(minRews, maxRews), rewSize=rewSize, userandom=True, reachable=s0, nRowExcl=4)
    for t in range(nTrials):
      trajs[e][t] = pachinko_sim.getPachinkoRandomTraj(mazes[e], s0, addAbsorbingSt=addAbsorbingSt)
  print()

generateMazeTrajs()

def hasReward(s, rewards):
  return s in rewards

def trainTrans(maze, trajs, trialsToPlot, addAbsorbingSt=False, rewardModScale=1):
  # learn a transition matrix for the given maze using the empirical trajectories
  # possibly with reward modulation (amount specified by rewardModScale)
  nRows, nCols = maze.shape
  nStates = maze.size
  rews = np.ravel_multi_index(np.where(maze > 0), (nRows, nCols))
  n = len(trialsToPlot)
  # init transition matrix
  T = np.zeros((nStates+addAbsorbingSt, nStates+addAbsorbingSt))
  Tlist = np.zeros((n, T.shape[0], T.shape[1]))
  # learn transition matrix based on trajs
  count = 0
  for j, traj in enumerate(trajs):
    stateSeq = [np.ravel_multi_index(s, (nRows, nCols)) if s[0] < nRows else nStates for s in traj]
    for i, prevState in enumerate(stateSeq[:-1]):
      curState = stateSeq[i+1]
      if hasReward(curState, rews):
        T[prevState, curState] += rewardModScale
      elif hasReward(prevState, rews) and rewardModScale != 1:
        T[prevState, curState] += 1
      else:
        T[prevState, curState] += 1
    if j+1 in trialsToPlot:
      Tlist[count,:] = T.copy()
      count += 1
  rowSums = Tlist.sum(axis=2)
  Tlist = np.nan_to_num(Tlist / rowSums[:,:,np.newaxis])
  return Tlist

# learn SR
nStates = nRows * nCols
Ms = np.zeros((len(gammas), nExps+1, len(trialsToPlot), nStates))
MTDs = np.zeros((len(gammas), nExps+1, len(trialsToPlot), nStates+addAbsorbingSt, nStates+addAbsorbingSt))
scheduler = schedulers[lrScheduleType]

def getSRfromTrans():
  print('Training...', end='')
  for i, gamma in enumerate(gammas):
    lr, rewardModScale = alpha, 1
    rewardModulateAlpha = rewardModulateAlphas[i]
    if rewardModulateAlpha: rewardModScale = 100
    for j, maze in enumerate(mazes):
      Tlist = trainTrans(maze, trajs[j], trialsToPlot, addAbsorbingSt=addAbsorbingSt, rewardModScale=rewardModScale)
      for k, _ in enumerate(trialsToPlot):
        Ts[i,j,k,:] = Tlist[k,:]
        MTDs[i,j,k,:], _, _ = bias_var.getSR(Tlist[k,:], gamma=gamma, hasAbsorbingSt=addAbsorbingSt, 
                                              extraTransOnSR=extraTransOnSR)
  print()

def trainSR():
  print('Training...', end='')
  for i, gamma in enumerate(gammas):
    lr = alpha
    rewardModulateAlpha = rewardModulateAlphas[i]
    for j, maze in enumerate(mazes):
      Minit = np.identity(nStates+addAbsorbingSt)
      count = 0
      trajl = trajs[j]
      if nExps >= 10 and (j+1) % (nExps//10) == 0: print(str((j+1) * 100 // nExps) + '%', end=' ')
      for k, traj in enumerate(trajl):
        # learning rate scheduler
        if lrScheduleType == 'step':
          if (k+1) % scheduler['epochs'] == 0: # apply step decay
            lr *= scheduler['rate']
            if rewardModulateAlpha: 
              rewardModulateAlpha *= scheduler['rate']
        elif lrScheduleType == 'exp':
          lr = alpha * np.exp(-scheduler['rate'] * k)
          if rewardModulateAlpha: 
            rewardModulateAlpha = rewardModulateAlphas[i] * np.exp(-scheduler['rate'] * k)
        MTD = bias_var.getTDSR(maze, traj, Minit, rewSize=rewSize, alpha=lr, gamma=gamma, lamb=lamb, addAbsorbingSt=addAbsorbingSt,
                                extraTransOnSR=extraTransOnSR, rewardThreshold=0, rewardModulateAlpha=rewardModulateAlpha)
        if k+1 in trialsToPlot:
          Ms[i,j,count,:] = MTD[stateIdx[s0],:-addAbsorbingSt]
          MTDs[i,j,count,:,:] = MTD
          count += 1
        Minit = MTD
  print()

trainSR()

sampCntList = [[] for i in range(len(gammas))]
vsamps = np.zeros((len(gammas), nExps, nSamples))
vtrues = np.zeros((len(gammas), nExps))

def getSampledValue():
  # estimate sample-based value
  print('Testing...', end='')
  for i, gamma in enumerate(gammas):
    for j, maze in enumerate(mazes[1:]):
      if nExps >= 10 and (j+1) % (nExps//10) == 0: print(str((j+1) * 100 // nExps) + '%', end=' ')
      M = MTDs[i,j+1,trueM_idx,:]
      Mext, _, _ = bias_var.getSR(trueT, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
      MTF_id = np.identity(M.shape[0])
      rewIdx = np.ravel_multi_index(np.where(maze == rewSize), (nRows, nCols))
      _, vtrue, vsamp, (rCnt, nrCnt), _ = bias_var.runSim(1, nSamples, nRows, nCols, rewIdx.size, 
                                                    s0, M, Mext, rewSize=rewSize, userandom=False, idx=rewIdx, MFC=MTF_id,
                                                    addAbsorbingSt=addAbsorbingSt, rho=rho, beta=beta, verbose=False)
      vtrues[i,j] = vtrue
      vsamps[i,j,:] = vsamp
      sampCntList[i].append((rCnt, nrCnt))
  print()

def getSampledValue2():
  # estimate sample-based value
  print('Testing...', end='')
  for i, gamma in enumerate(gammas):
    for j, maze in enumerate(mazes[1:]):
      if nExps >= 10 and (j+1) % (nExps//10) == 0: print(str((j+1) * 100 // nExps) + '%', end=' ')
      M = MTDs[i,j+1,trueM_idx,:]
      Mext = MTDs[0,j+1,trueM_idx,:]
      # Mext, _, _ = bias_var.getSR(trueT, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=extraTransOnSR)
      MTF_id = np.identity(M.shape[0])
      rewIdx = np.ravel_multi_index(np.where(maze > 0), (nRows, nCols))
      _, vtrue, vsamp, (rCnt, nrCnt), _ = bias_var.runSim2(1, nSamples, nRows, nCols, rewIdx.size, maze,
                                                            s0, M, Mext, MFC=MTF_id, addAbsorbingSt=addAbsorbingSt, 
                                                            rho=rho, beta=beta, verbose=False, 
                                                            mod=rewardModulateAlphas[i], modFrac=0)
      vtrues[i,j] = vtrue
      vsamps[i,j,:] = vsamp
      sampCntList[i].append((rCnt, nrCnt))
  print()

getSampledValue2()

# plot sample distribution in terms of rewarding vs non-rewarding states
def plotSampleRewardFrac(sampCntList, rewardModulateAlphas):
  nsamps = [10,100,1000]
  fig, axes = plt.subplots(len(rewardModulateAlphas), 3, figsize=[6.6*cm, 4.4*cm], dpi=300)
  for i, gamma in enumerate(rewardModulateAlphas):
    l = sampCntList[i]
    rCntAvg, nrCntAvg = np.array([rCnt.mean(axis=0) for (rCnt, nrCnt) in l]), np.array([nrCnt.mean(axis=0) for (rCnt, nrCnt) in l])
    total = rCntAvg + nrCntAvg
    rCntAvg, nrCntAvg = rCntAvg / total, nrCntAvg / total
    height = np.array([rCntAvg.mean(axis=0), nrCntAvg.mean(axis=0)])
    errs = np.array([sem(rCntAvg), sem(nrCntAvg)])
    for j in range(height.shape[1]):
      axes[i,j].bar([0,1], height[:,j], tick_label=['r.', 'n.r.'], yerr=errs[:,j],
                    error_kw=dict(lw=1), color=['m', 'gray'])
      axes[i,j].set_ylim([0, 1])
      axes[i,j].set_ylabel('average reward\nfraction', fontdict={'fontsize':5}, labelpad=1)
      axes[i,j].tick_params(labelsize=5, length=2, pad=1)
      plt.setp(axes[i,j].spines.values(), linewidth=0.5)
      if gamma:
        axes[i,j].set_title(r'$\alpha_{mod}=$' + str(gamma), size=5, pad=2)
      else:
        pass
        # axes[i,j].set_title('no reward modulation', size=7, pad=5)
  plt.tight_layout(pad=0.25)
  plt.show()

plotSampleRewardFrac(sampCntList, rewardModulateAlphas)

# plot bias-variance tradeoff
bias_var.plotValueEstLearnedSR(nSamples, vtrues, vsamps, beta, gammas=gammas, rewardModulateAlphas=rewardModulateAlphas, xabs_max=.2)

def plotLearningExample():
  # plot boards & learned SR over time
  vmin, vmax = [0] * len(gammas), [None] * len(gammas)
  vmax_alt = 0.01
  assert(len(gammas) == len(rewardModulateAlphas))
  for i, rma in enumerate(rewardModulateAlphas):
    if rma: vmax[i] = .5
    else: vmax[i] = .5

  demoIdx = 0
  s0Idx = np.ravel_multi_index(s0, (nRows, nCols))
  maze = mazes[demoIdx]
  tjs = trajs[demoIdx]

  fig = plt.figure(1, figsize=[18*cm, 5*cm], dpi=300)
  ims = []
  for ti, t in enumerate(trialsToPlot):
    traj = tjs[t-1]
    for i, gamma in enumerate(gammas):
      rewardModulateAlpha = rewardModulateAlphas[i]
      # plot the board with traj and rewards marked
      ax0 = plt.subplot(len(gammas),len(trialsToPlot)*2,i*len(trialsToPlot)*2+ti*2+1)
      for s in traj[:-addAbsorbingSt]:
        ax0.plot(s[1]+0.5, nRows-s[0]-0.5, 'kx', markersize=4, mew=0.5)
      rewR, rewC = np.where(maze > 0)
      for j, r in enumerate(rewR):
        c = rewC[j]
        if i == 0 and j == 0: ax0.plot(c+0.5, nRows-r-0.5, 'm*', markersize=4, mew=0.5, label='Reward')
        else: ax0.plot(c+0.5, nRows-r-0.5, 'm*', markersize=4, mew=0.5)
      if t < 5: ax0.set_title('Trial {}'.format(t), size=5, pad=2)
      ax0.set_xticks(np.arange(0, nCols+1, 1))
      ax0.set_yticks(np.arange(0, nRows, 1))
      ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
      ax0.grid()
      ax0.set_aspect('equal', adjustable='box')
      # plot the learned SR
      ax1 = plt.subplot(len(gammas),len(trialsToPlot)*2,i*len(trialsToPlot)*2+ti*2+2)
      sr = MTDs[i,demoIdx,ti,s0Idx,:-addAbsorbingSt].reshape(maze.shape)
      # constrain all values to the prespecified range for plotting
      if t < 10:
        sr = np.clip(sr, vmin[i], vmax_alt)
        im = ax1.imshow(sr, vmin=vmin[i], vmax=vmax_alt, cmap="Greys")
      else:
        sr = np.clip(sr, vmin[i], vmax[i])
        im = ax1.imshow(sr, vmin=vmin[i], vmax=vmax[i], cmap="Greys")
      if ti == 0 or ti == len(trialsToPlot)-1: ims.append(im)
      modulateStr = r'$\alpha_{mod}=$' + str(rewardModulateAlpha) if rewardModulateAlpha else r'$\alpha=$' + str(alpha)
      ax1.set_title(modulateStr, size=5, pad=2)
      ax1.set_xticks(np.arange(-.5, nCols, 1))
      ax1.set_yticks(np.arange(-.5, nRows, 1))
      ax1.set_xticklabels(np.arange(0, nCols+1, 1))
      ax1.set_yticklabels(np.arange(0, nRows+1, 1))
      ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
      ax1.grid()
      ax1.set_aspect('equal', adjustable='box')
  # ax0.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)

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
  # plt.tight_layout()
  plt.savefig('f5-traj.pdf', dpi=300)
  plt.show()

plotLearningExample()
