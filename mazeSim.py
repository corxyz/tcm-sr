import numpy as np
from numpy.random import choice, rand
import pylab as plt

import sys
np.set_printoptions(threshold=sys.maxsize)

def initMaze(rewFrac=0.1, rewSize=1, shape='8'):
  if shape == '8':
    # generate a 8-shaped maze
    nRows, nCols = 6, 11
    maze = np.zeros([nRows, nCols])
    maze[1:5, 1:5] = -1000000
    maze[1:5, -5:-1] = -1000000
    nRews = round((maze.size - np.count_nonzero(maze)) * rewFrac)
    # place rewards on the maze
    freeR, freeC = np.where(maze == 0)
    rewIdx = choice(freeR.size, nRews)
    maze[freeR[rewIdx], freeC[rewIdx]] = rewSize
  return maze

def initTrans(maze, addAbsorbingSt=True):
  # can only move right or down
  nRows, nCols = maze.shape
  nStates = maze.size
  if addAbsorbingSt: 
    T = np.zeros([nStates+1, nStates+1])
    T[nStates, nStates] = 1
    T[nStates-1, nStates] = 1
  else: T = np.zeros([nStates, nStates])
  for s in range(nStates):
    i, j = np.unravel_index(s, (nRows, nCols))
    if maze[i,j] < 0: continue      # what to do with the blocked states?
    neighbors = []
    if i < nRows - 1 and maze[i+1,j] >= 0: neighbors.append((i+1,j))
    if j < nCols - 1 and maze[i,j+1] >= 0: neighbors.append((i,j+1))
    for neighbor in neighbors:
      T[s, np.ravel_multi_index(neighbor, (nRows, nCols))] = 1/len(neighbors)
  return T

def getSR(T, gamma=1, extraTransOnSR=True):
  if extraTransOnSR:
    M = np.matmul(np.linalg.inv(np.identity(T.shape[0]) - gamma*T), T)
  else:
    M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  return M

def plotMazeSR(maze, s0, gamma=1):
  T = initTrans(maze)
  M = getSR(T, gamma=gamma) 
  plt.figure()
  plt.subplot(2,1,1)
  plt.imshow(maze, cmap="Greys")
  plt.colorbar()
  plt.subplot(2,1,2)
  plt.imshow(np.reshape(M[s0,:], maze.shape), cmap="Greys")
  plt.colorbar()
  plt.show()

def runSim(nExps, nSamples, s0, M, M_gamma1, shape='8',
            rewFrac=0.1, rewSize=1, userandom=True, idx=[2,8,14,27,34,49,52,65,79,87],
            MFC=None, addAbsorbingSt=True, 
            rho=None, beta=None, ensureCnormIs1=True):
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(nExps)
  vgamma1 = np.zeros(nExps)
  vsamp = np.zeros((nExps,nSamples))
  sampRow = np.zeros((nExps,nSamples), dtype=int)
  sampCol = np.zeros((nExps,nSamples), dtype=int)

  # compute rho and beta (if one of them was not provided aka None)
  if rho == None:
    rho = 1-beta
  elif beta == None:
    beta = 1-rho

  for e in range(nExps):
    if (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
    maze = initMaze(rewFrac=rewFrac, rewSize=rewSize, shape=shape)
    nRows, nCols = maze.shape
    rvec = maze.flatten() # 1-step rewards
    if addAbsorbingSt: rvec = np.append(rvec, 0)
    nStates = maze.size
    stateIdx = np.arange(0, nStates).reshape(maze.shape)    # state indices
    stim = np.identity(nStates + addAbsorbingSt)

    vtrue[e] = np.dot(M[stateIdx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    vgamma1[e] = np.dot(M_gamma1[stateIdx[s0],:], rvec)       # Compute the actual value function (as v=Mr)

    c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)

    sampleIdx = np.negative(np.ones(nSamples, dtype=int))
    for i in range(nSamples):
      # define sampling distribution
      a = np.matmul(M.T,c)
      P = a / np.sum(a)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(P))[0]
      sampleIdx[i] = tmp[0]
      if sampleIdx[i] >= nRows * nCols: # sampled absorbing state:
        f = stim[:,stateIdx[s0]]               # update context with starting state rather than absorbing state
        break
      else:
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
    rewsamp = np.append(rewsamp, np.zeros(nSamples - len(rewsamp)))
    vsamp[e,:] = rewsamp * np.sum(M[stateIdx[s0],:]) # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)    

  return vtrue, vgamma1, vsamp, sampRow

def plotSampleDistGammas(maze, s0, nExps, sampRowList, shape='8',
                         rewIdx=None, gammas=None, rho=None, beta=None, rewardModulateAlphas=None):
  assert(len(sampRowList) == len(gammas))
  fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})

  # plot the maze with starting position marked
  nRows, nCols = maze.shape
  maze = np.clip(maze, -1, 0)
  ax0.imshow(maze, cmap="Greys_r")
  ax0.plot(s0[1], s0[0], 'c*', markersize=20)
  if rewIdx:
    for idx in rewIdx:
      r, c = np.unravel_index(idx, (nRows, nCols))
      ax0.plot(c+0.5, nRows-r-0.5, 'r*', markersize=20)
  ax0.set_xticks(np.arange(-.5, nCols, 1))
  ax0.set_yticks(np.arange(-.5, nRows, 1))
  ax0.set_xticklabels(np.arange(0, nCols+1, 1))
  ax0.set_yticklabels(np.arange(0, nRows+1, 1))
  ax0.tick_params(length=0, labelbottom=True, labelleft=True)   
  ax0.grid()
  ax0.set_aspect('equal', adjustable='box')

  # plot the sampling distributions of different gammas
  y = np.arange(0, nRows, 1)
  c = ['r','b','g']
  ax1.invert_yaxis()
  for i, gamma in enumerate(gammas):
    d = sampRowList[i].flatten()
    d = d[d >= 0]
    cnt = np.array([(d == r).sum()/float(nExps) for r in y])
    psamp = cnt / cnt.sum()
    theo = np.array([(1-gamma) * gamma ** (r-1) if r > 0 else 0 for r in y])
    theo = theo / theo.sum()
    
    modAlpha = None
    if rewardModulateAlphas: modAlpha = rewardModulateAlphas[i]
    if modAlpha:
      ax1.plot(psamp, y, c[i%len(c)] + 'x', 
                label=r'$\gamma=$' + str(gamma) + ', ' + r'$\alpha_{mod}=$' + str(modAlpha) + ' (empirical)')
    else:
      ax1.plot(theo, y, c[i%len(c)] + '-', linewidth=0.5, label=r'$\gamma=$' + str(gamma) + ' (expected)')
      ax1.plot(psamp, y, c[i%len(c)] + 'x', label=r'$\gamma=$' + str(gamma) + ' (empirical)')
    ax1.set_xticks(np.linspace(0, round(max(np.nanmax(theo), np.nanmax(psamp)),1), num=5))
    ax1.set_yticks(np.arange(0, nRows, 1))
    ax1.set_xlabel(r'$P(i_t)$')
    ax1.set_ylabel('Row number')

  fig.subplots_adjust(right=0.8)
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left')
  plt.tight_layout()
  plt.show()
