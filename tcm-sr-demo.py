# Basic TCM-SR code

import numpy as np
from numpy.random import randn

from typing import List, Tuple, Optional

def initMaze(numRows: int, numCols: int, nRews: int, 
              rewSize: float=1.0, userandom: bool=False, idx: List[int]=[2,8,14,27,34,49,52,65,79,87]) -> np.ndarray:
  '''
  Params:
    numRows: number of rows in the maze
    numCols: number of columns in the maze
    nRews: number of rewards in the maze
    rewSize: reward size
    userandom: use random reward placement
    idx: a list of pre-determined reward positions
  
  Return:
    maze: a maze of specified dims with rewards
  '''
  maze = np.zeros([numRows, numCols])
  if userandom:   # use random reward placement, avoiding the first row
    p = np.random.permutation((numRows-1) * numCols)
    rewIdx = p[:nRews]    # reward locations
    rewIdx = np.unravel_index(rewIdx, (numRows, numCols))
    # place rewards on the maze
    mazeMiddle = np.zeros([numRows-1,numCols])
    mazeMiddle[rewIdx] = rewSize
    maze[1:,:] =  mazeMiddle  # start placing reward from the second row
  else:           # use the first nRews in the provided reward locations (idx)
    rewIdx = np.unravel_index(idx[:min(len(idx),nRews)], (numRows, numCols))
    maze[rewIdx] = rewSize
  return maze


def initTrans(numRows: int, numCols: int, addAbsorbingSt: bool=False, reversible: bool=False) -> np.ndarray:
  '''
  Computes the transition matrix of a maze where only diagonal movements are permitted

  Params:
    numRows: number of rows in the maze
    numCols: number of columns in the maze
    addAbsorbingSt: add an absorbing state at the end of the maze
    reversible: if true, both upward and downward transitions are possible;
                if false, only downward transitions are allowed (i.e. standard Pachinko)
  
  Return:
    T: one-step transition matrix
  '''
  nStates = numRows * numCols
  # init transition matrix
  if addAbsorbingSt: 
    T = np.zeros([nStates+1, nStates+1])
    T[nStates, nStates] = 1
  else: T = np.zeros([nStates, nStates])

  for s in range(nStates):
    # for each position, obtain a list of positions reachable in one step (neighbors)
    # then fill in transition probabilities assuming equal probability of visiting any neighbor in one step
    i, j = np.unravel_index(s, (numRows, numCols))
    neighbors = []
    if i < numRows-1:   # not at the bottom
      if j > 0 and j < numCols-1: # not at the walls
        neighbors += [(i+1, j-1), (i+1, j+1)]
        if reversible and i > 0: neighbors += [(i-1, j-1), (i-1, j+1)]
      elif j == 0:      # at the left wall
        neighbors.append((i+1, j+1))
        if reversible and i > 0: neighbors.append((i-1, j+1))
      else:             # at the right wall
        neighbors.append((i+1, j-1))
        if reversible and i > 0: neighbors.append((i-1, j-1))
      for neighbor in neighbors:
        T[s, np.ravel_multi_index(neighbor, (nRows, nCols))] = 1/len(neighbors)
    else:               # at the bottom
      if addAbsorbingSt: # enter the absorbing state if there is one
        T[s, nStates] = 1
      else:              # otherwise stay put at the bottom
        T[s, s] = 1
  return T


def getSR(T: np.ndarray, gamma: float=1.0, extraTransOnSR: bool=False) -> np.ndarray:
  '''
  Computes the successor representation matrix for the given one-step transition function and gamma

  Param:
    T: one-step transition matrix
    gamma: discount factor
    extraTransOnSR: if true, include an extra transition at each step to avoid stalling at the current location
  
  Return:
    M: successor representation of the specified gamma
  '''
  if extraTransOnSR:  # include an extra transition at each step
    M = np.matmul(np.linalg.inv(np.identity(T.shape[0]) - gamma*T), T)
  else:
    M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  return M


def runSim(nExps: int, nSamples: int, nRows: int, nCols: int, nRews :int, s0: Tuple[int, int], M: np.ndarray,
            rewSize: float=1.0, userandom: bool=True, idx: List[int]=[2,8,14,27,34,49,52,65,79,87], randomRewards: bool=False,
            MFC: Optional[np.ndarray]=None, addAbsorbingSt: bool=True,
            rho: float=1.0, beta: float=0.0, ensureCnormIs1: bool=True, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  Params:
    nExps: number of experiments/mazes
    nSamples: number of samples drawn per experiment
    nRows: number of rows in each maze
    nCols: number of cols in each maze
    s0: starting position (same across experiments)
    M: SR matrix
    rewSize: reward size
    userandom: use random reward placement
    idx: a list of pre-determined reward positions
    randomRewards: random reward size for each maze position
    MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
    addAbsorbingSt: add an absorbing state at the end of the maze (make sure dim of M/MFC includes this if used)
    rho: TCM param, specifies how much of the previous context vector to retain
    beta: TCM param, specifies how much of the new incoming context vector to incorporate
    ensureCnormIs1: if true, checks the norm of the updated context vector is 1
    verbose: if true, prints progress (% complete)
  
  Return:
    vtrue: true value of s0 in each experiment
    vsamp: sample-based estimated value of s0 in each experiment
    sampRow: sampled rows
  '''
  vtrue = np.zeros(nExps)
  vsamp = np.zeros((nExps,nSamples))
  sampRow = np.zeros((nExps,nSamples), dtype=int)
  sampCol = np.zeros((nExps,nSamples), dtype=int)

  if not MFC:
    MFC = np.identity(M.shape[0])

  for e in range(nExps):
    if verbose and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
    maze = initMaze(nRows, nCols, nRews, rewSize=rewSize, userandom=userandom, idx=idx)
    rvec = maze.flatten() # one-step rewards
    if randomRewards: rvec = 1 + randn(rvec.size)
    if addAbsorbingSt: rvec = np.append(rvec, 0)                  # no reward in the absorbing state

    stateIdx = np.arange(0, nRows * nCols).reshape(maze.shape)    # state indices
    stim = np.identity(nRows * nCols + addAbsorbingSt)

    vtrue[e] = np.dot(M[stateIdx[s0],:], rvec)                    # Compute the actual value function (as v=Mr)

    c = stim[:,stateIdx[s0]]                                      # starting context vector (set to the starting state)

    sampleIdx = np.negative(np.ones(nSamples, dtype=int))
    for i in range(nSamples):
      # define sampling distribution
      a = np.matmul(M.T,c)
      P = a / np.sum(a)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(P))[0]
      sampleIdx[i] = tmp[0]
      if sampleIdx[i] >= nRows * nCols:     # sampled absorbing state:
        f = stim[:,stateIdx[s0[0],s0[1]]]   # update context with starting state rather than absorbing state 
      else:
        sampRow[e,i], sampCol[e,i] = np.unravel_index(sampleIdx[i], (nRows, nCols))
        f = stim[:,sampleIdx[i]]

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN         # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if ensureCnormIs1:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[sampleIdx[sampleIdx >= 0]]
    rewsamp = np.append(rewsamp, np.zeros(nSamples - len(rewsamp)))
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[stateIdx[s0],:]) 

  return vtrue, vsamp, sampRow
