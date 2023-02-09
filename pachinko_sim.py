import numpy as np
from numpy.random import rand, randint, uniform, randn
import pylab as plt
import math

cm = 1/2.54  # centimeters in inches

def getReachableStates(nRows, nCols, nRowExcl, s0):
  T = initTrans(nRows * nCols, nRows, nCols)
  M, _ = getSR(T, gamma=0.99999)
  start = np.ravel_multi_index(s0, (nRows, nCols))
  reachables = np.nonzero(M[start, :(nRows-nRowExcl+1)*nCols])[0]
  return list(reachables[1:])

def initMaze(nRows, nCols, nRews, rewSize=1, 
              userandom=False, idx=[2,8,14,27,34,49,52,65,79,87], reachable=None, nRowExcl=4,
              forceRow=None):
  assert(nRews > 0)
  # reachable could be (i) None (ii) a tuple indicating the state position (iii) a list of tuples (states)
  maze = np.zeros([nRows, nCols])
  if userandom:
    if reachable:
      allReachables = []
      if type(reachable) is tuple:
        allReachables = getReachableStates(nRows, nCols, nRowExcl, reachable)
      elif type(reachable) is list:
        for s0 in reachable:
          allReachables += getReachableStates(nRows, nCols, nRowExcl, s0)
        allReachables = list(set(allReachables))
      p = np.random.permutation(allReachables)

      rewIdx = p[:nRews]
      if forceRow: # force at least one reward to be placed in the row specified
        for pos in p: 
          if forceRow * nCols <= pos < (forceRow+1) * nCols:
            rewIdx[0] = pos
            break
      rewIdx = np.unravel_index(rewIdx, (nRows, nCols))
      # Place rewards on the maze
      maze[rewIdx] =  rewSize
    else:
      p = np.random.permutation((nRows - nRowExcl) * nCols)
      rewIdx = p[:nRews]
      rewIdx = np.unravel_index(rewIdx, (nRows, nCols))
      # Place rewards on the maze (avoiding first and nRowExcl-1 last rows)
      mazeMiddle = np.zeros([nRows-nRowExcl,nCols])
      mazeMiddle[rewIdx] = rewSize
      if nRowExcl > 1: maze[1:-nRowExcl+1,:] = mazeMiddle
      else: maze[1:,:] =  mazeMiddle
  else:
    rewIdx = np.unravel_index(idx[:min(len(idx),nRews)], (nRows, nCols))
    maze[rewIdx] = rewSize
  return maze

def initTrans(nStates, nRows, nCols, addAbsorbingSt=False, reversible=False):
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

def randomStepPachinko(nRows, nCols, curRow, curCol, addAbsorbingSt=False):
  # take one step in a random direction in the pachinko
  directions = []
  if curRow == nRows - 1: 
    if addAbsorbingSt:  # move to absorbing state; otherwise stay at the same place
      curRow, curCol = nRows, 0
    return curRow, curCol
  elif 0 < curCol < nCols - 1: directions = [(1,1),(1,-1)]
  elif curCol == 0: directions = [(1,1)]
  else: directions = [(1,-1)]
  step = directions[randint(len(directions))]
  curRow += step[0]
  curCol += step[1]
  return curRow, curCol

def getPachinkoRandomTraj(maze, s0, addAbsorbingSt=True):
  # run one random Pachinko game (number of steps is equal to the number of rows - 1)
  nRows, nCols = maze.shape
  curRow, curCol = s0
  traj = [(curRow, curCol)]           # keep track of the trajectory
  while curRow < nRows:
    curRow, curCol = randomStepPachinko(nRows, nCols, curRow, curCol, addAbsorbingSt=addAbsorbingSt)   # take one random step in the maze
    traj.append((curRow, curCol))
  return traj

def getSR(T, gamma=1, hasAbsorbingSt=False, extraTransOnSR=False):
  if extraTransOnSR:
    M = np.matmul(np.linalg.inv(np.identity(T.shape[0]) - gamma*T), T)
  else:
    M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  if hasAbsorbingSt:
    M = M[:-1,:-1]
    T = T[:-1,:-1]
  return M, T

def drawSamples(s0, maze, stateIdx, stim, M, nSamples, rho=None, beta=None, addAbsorbingSt=True, MFC=np.identity(0),
                ensureCnormIs1=True, plotSamples=True, inclLastSamp=False):
  if MFC.shape != M.shape:
    MFC = np.identity(M.shape[0]) # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  if addAbsorbingSt:
    M = M[:-1, :-1]
    MFC = MFC[:-1, :-1]
    stim = stim[:-1, :-1]

  c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)
  rvec = maze[:]

  sampleIdx = np.negative(np.ones(nSamples, dtype=int))
  nRows, nCols = maze.shape
  nRowPlot, nColPlot = 1, 4
  fig = plt.figure(figsize=[8.8*cm, 3*cm], dpi=300)
  for i in range(nSamples):
    # define sampling distribution
    a = np.matmul(M.T, c)
    P = a / np.sum(a)
    assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

    # draw sample
    tmp = np.where(rand() <= np.cumsum(P))[0]
    sampleIdx[i] = tmp[0]
    f = stim[:,sampleIdx[i]]        # sample vector
    cIN = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
    cIN = cIN/np.linalg.norm(cIN)

    # update context
    c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
    c = c/np.linalg.norm(c)
    if ensureCnormIs1:
      assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # plot
    if plotSamples and i < nRowPlot * nColPlot:
      axes = plt.subplot(nRowPlot, nColPlot, i+1)
      # draw the board
      title = r'$\mathbf{x}$'
      axes.set_title(title + r'$_{}$'.format(str(i+1)), size=7, pad=2)
      axes.set_xticks(np.arange(-.5, nCols, 1))
      axes.set_yticks(np.arange(-.5, nRows, 1))
      axes.tick_params(length=0, labelbottom=False, labelleft=False)   
      axes.grid()
      axes.set_aspect('equal', adjustable='box')
      # plot sampling distribution
      im = plt.imshow(np.reshape(P, maze.shape), cmap="Greys")

      # plot sample
      sRow,sCol = np.unravel_index(sampleIdx[i], maze.shape)
      axes.plot(sCol, sRow, 'c*', markersize=5, label="sample")

      if inclLastSamp:
        # mark previous sample
        if i==0:
          prevRow, prevCol = s0[0], s0[1]
        else:
          prevRow, prevCol = np.unravel_index(sampleIdx[i-1], maze.shape)
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

  return sampleIdx

def plotSampleDist(s0, nRows, nCols, nExps, sampRow, gamma=0, rho=None, beta=None, steps=3):
  fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})

  # plot the board with starting position marked
  ax0.plot(s0[1]+0.5, nRows-0.5, 'c*', markersize=20)
  ax0.set_xticks(np.arange(0, nCols+1, 1))
  ax0.set_yticks(np.arange(0, nRows, 1))
  ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax0.grid()
  ax0.set_aspect('equal', adjustable='box')

  # plot the sampling distributions of the context at the first three time steps
  y = np.arange(0, nRows, 1)
  ax1.invert_yaxis()
  d = sampRow[:,:steps].flatten()
  d = d[d >= 0]
  cnt = np.array([(d == r).sum()/float(nExps) for r in y])
  psamp = cnt / cnt.sum()
  psamp[0] = np.nan
  theo = np.array([np.array([(1-gamma) * gamma ** (r-offset) if r-offset >= 0 else np.nan for r in y]) for offset in range(1, 4)])
  theo = theo / np.nansum(theo, axis=1)[:, np.newaxis]
  ax1.plot(theo.T, np.tile(y, (steps,1)).T, linewidth=0.5, label=r'$\gamma={} (expected)$'.format(gamma))
  ax1.plot(psamp, y, 'bx', label=r'$\gamma={} (empirical)$'.format(gamma))
  ax1.set_xticks(np.arange(0, 1, .25))
  ax1.set_yticks(np.arange(0, nRows, 1))
  ax1.set_xlabel(r'$P(i_t)$')
  ax1.set_ylabel('Row number')

  fig.subplots_adjust(right=0.8)
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left')
  plt.tight_layout()
  plt.show()

def plotSampleDistGammas(s0, nRows, nCols, nExps, sampRowList, 
                         rewIdx=None, gammas=None, rho=None, beta=None, rewardModulateAlphas=None):
  assert(len(sampRowList) == len(gammas))
  fig, axes = plt.subplots(len(gammas), 1, figsize=[3*cm,2.8*len(gammas)*cm], dpi=300)

  # plot the board with starting position marked
  # ax0.plot(s0[1]+0.5, nRows-0.5, 'c*', markersize=20)
  # if rewIdx:
  #   for idx in rewIdx:
  #     r, c = np.unravel_index(idx, (nRows, nCols))
  #     ax0.plot(c+0.5, nRows-r-0.5, 'r*', markersize=20)
  # ax0.set_xticks(np.arange(0, nCols+1, 1))
  # ax0.set_yticks(np.arange(0, nRows+1, 1))
  # ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
  # ax0.grid()
  # ax0.set_aspect('equal', adjustable='box')

  # plot the sampling distributions of different gammas
  y = np.arange(0, nRows, 1)
  c = 'g'
  for i, gamma in enumerate(gammas):
    ax = axes[i]
    ax.invert_yaxis()
    d = sampRowList[i].flatten()
    d = d[d >= 0]
    cnt = np.array([(d == r).sum()/float(nExps) if r > 0 else 0 for r in y])
    psamp = cnt / cnt.sum()
    psamp[0] = np.nan
    theo = np.array([(1-gamma) * gamma ** (r-1) if r > 0 else 0 for r in y])
    theo = theo / theo.sum()
    theo[0] = np.nan
    
    modAlpha = None
    if rewardModulateAlphas: modAlpha = rewardModulateAlphas[i]
    if modAlpha:
      ax.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.5,
                label='empirical (' + r'$\alpha_{mod}=$' + str(modAlpha) + ')')
    else:
      ax.plot(theo, y, c[i%len(c)] + '-', linewidth=0.5, label='expected')
      ax.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.8, label='empirical')
    ax.set_xticks(np.linspace(0, 1, num=5))
    ax.set_yticks(np.arange(0, nRows, 1))
    ax.set_xlabel(r'$P(i_t)$', fontdict={'fontsize':5}, labelpad=2)
    ax.set_ylabel('Row number', fontdict={'fontsize':5}, labelpad=2)
    ax.tick_params(labelsize=5, length=2, pad=1)
    # ax.legend(fontsize=5, loc='lower right')

  # fig.subplots_adjust(top=0.95, bottom=0.55, left=0.3)
  plt.tight_layout(pad=0.25)
  plt.show()

def plotTrans(maze, T, power=1):
  nRows, nCols = maze.shape
  plt.figure()
  plt.set_xticks(np.arange(-.5, nCols, 1))
  plt.set_yticks(np.arange(-.5, nRows, 1))
  plt.set_xticklabels(np.arange(0, nCols+1, 1))
  plt.set_yticklabels(np.arange(0, nRows+1, 1))
  plt.tick_params(length=0, labelbottom=False, labelleft=False)   
  plt.grid()
  plt.imshow(matrix_power(T, power), cmap="Greys")

  plt.colorbar()
  plt.show()

def plotSR(maze, M, s0, stateIdx, addAbsorbingSt=False):
  nRows, nCols = maze.shape
  fig, ax = plt.subplots()
  # plot the board with starting position marked
  ax.set_xticks(np.arange(-.5, nCols, 1))
  ax.set_yticks(np.arange(-.5, nRows, 1))
  ax.set_xticklabels(np.arange(0, nCols+1, 1))
  ax.set_yticklabels(np.arange(0, nRows+1, 1))
  ax.tick_params(length=0, labelbottom=False, labelleft=False) 
  ax.grid()
  im = ax.imshow(np.reshape(M[stateIdx[s0],:-addAbsorbingSt], (nRows, nCols)), cmap="Greys")

  plt.colorbar(im)
  plt.tight_layout()
  plt.show()

def plotMaze(maze, s0):
  nRows, nCols = maze.shape
  fig = plt.figure(figsize=[2.2*cm, 3*cm], dpi=300)
  # plot the board with starting position marked
  rewR, rewC = np.where(maze > 0)
  plt.xticks(np.arange(-.5, nCols, 1))
  plt.yticks(np.arange(-.5, nRows, 1))
  plt.tick_params(length=0, labelbottom=False, labelleft=False)   
  plt.grid()
  plt.axis('scaled')

  # plt.plot(s0[1], nRows-s0[0]-1, 'ro', markersize=4, mew=0.5)
  # for j, r in enumerate(rewR):
  #   c = rewC[j]
  #   plt.plot(c, nRows-r-1, 'm*', markersize=4, mew=0.5)

  plt.show()

def plotTransSR(nRows, nCols, stateIdx, s0, T, M, addAbsorbingSt=False):
  powerT = T.copy()
  fig = plt.figure(1, figsize=[15*cm, 5*cm], dpi=300)
  # plot transitions matrices
  for i in range(1,nRows):
    ax0 = plt.subplot(1,nRows,i)
    t = np.reshape(powerT[stateIdx[s0],:-addAbsorbingSt], (nRows, nCols))
    im = ax0.imshow(t, vmin=0, vmax=0.5, cmap="Greys")
    ax0.set_title(r'$T^{}$'.format(i), size=5)
    ax0.set_xticks(np.arange(-.5, nCols, 1))
    ax0.set_yticks(np.arange(-.5, nRows, 1))
    ax0.set_xticklabels(np.arange(0, nCols+1, 1))
    ax0.set_yticklabels(np.arange(0, nRows+1, 1))
    ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
    ax0.grid()
    ax0.set_aspect('equal', adjustable='box')
    powerT = np.matmul(powerT, T)
  cbar_ax = fig.add_axes([0.04, 0.36, 0.01, 0.275])
  cbar = plt.colorbar(im, cax=cbar_ax, ticks=[0, 0.5])
  cbar.ax.tick_params(labelsize=5, pad=2)
  
  # plot SR
  ax1 = plt.subplot(1,nRows, nRows)
  sr = np.reshape(M[stateIdx[s0],:-addAbsorbingSt], (nRows, nCols))
  im = ax1.imshow(sr, cmap="Greys")
  ax1.set_title(r'$M$', size=5)
  ax1.set_xticks(np.arange(-.5, nCols, 1))
  ax1.set_yticks(np.arange(-.5, nRows, 1))
  ax1.set_xticklabels(np.arange(0, nCols+1, 1))
  ax1.set_yticklabels(np.arange(0, nRows+1, 1))
  ax1.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax1.grid()
  ax1.set_aspect('equal', adjustable='box')

  plt.show()
