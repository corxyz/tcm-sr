# Basic TCM-SR code
# Can be ran as a standalone demo

from typing import List, Tuple, Optional
import numpy as np
from numpy.random import randn, rand

def init_maze(n_row: int, n_col: int, n_rew: int, 
              rew_size: float=1.0, userandom: bool=False, idx: List[int]=[2,8,14,27,34,49,52,65,79,87]) -> np.ndarray:
  '''
  Params:
    n_row: number of rows in the maze
    n_col: number of columns in the maze
    n_rew: number of rewards in the maze
    rew_size: reward size
    userandom: use random reward placement
    idx: a list of pre-determined reward positions
  
  Return:
    maze: a maze of specified dims with rewards
  '''
  maze = np.zeros([n_row, n_col])
  if userandom:   # use random reward placement, avoiding the first row
    p = np.random.permutation((n_row-1) * n_col)
    rew_idx = p[:n_rew]    # reward locations
    rew_idx = np.unravel_index(rew_idx, (n_row, n_col))
    # place rewards on the maze
    maze_middle = np.zeros([n_row-1,n_col])
    maze_middle[rew_idx] = rew_size
    maze[1:,:] =  maze_middle  # start placing reward from the second row
  else:           # use the first n_rew in the provided reward locations (idx)
    rew_idx = np.unravel_index(idx[:min(len(idx),n_rew)], (n_row, n_col))
    maze[rew_idx] = rew_size
  return maze


def init_trans(n_row: int, n_col: int, add_absorb_state: bool=False, reversible: bool=False) -> np.ndarray:
  '''
  Computes the transition matrix of a maze where only diagonal movements are permitted

  Params:
    n_row: number of rows in the maze
    n_col: number of columns in the maze
    add_absorb_state: add an absorbing state at the end of the maze
    reversible: if true, both upward and downward transitions are possible;
                if false, only downward transitions are allowed (i.e. standard Pachinko)
  
  Return:
    T: one-step transition matrix
  '''
  n_state = n_row * n_col
  # init transition matrix
  if add_absorb_state: 
    T = np.zeros([n_state+1, n_state+1])
    T[n_state, n_state] = 1
  else: T = np.zeros([n_state, n_state])

  for s in range(n_state):
    # for each position, obtain a list of positions reachable in one step (neighbors)
    # then fill in transition probabilities assuming equal probability of visiting any neighbor in one step
    i, j = np.unravel_index(s, (n_row, n_col))
    neighbors = []
    if i < n_row-1:   # not at the bottom
      if j > 0 and j < n_col-1: # not at the walls
        neighbors += [(i+1, j-1), (i+1, j+1)]
        if reversible and i > 0: neighbors += [(i-1, j-1), (i-1, j+1)]
      elif j == 0:      # at the left wall
        neighbors.append((i+1, j+1))
        if reversible and i > 0: neighbors.append((i-1, j+1))
      else:             # at the right wall
        neighbors.append((i+1, j-1))
        if reversible and i > 0: neighbors.append((i-1, j-1))
      for neighbor in neighbors:
        T[s, np.ravel_multi_index(neighbor, (n_row, n_col))] = 1/len(neighbors)
    else:               # at the bottom
      if add_absorb_state: # enter the absorbing state if there is one
        T[s, n_state] = 1
      else:              # otherwise stay put at the bottom
        T[s, s] = 1
  return T


def get_sr(T: np.ndarray, gamma: float=1.0, extra_trans_on_sr: bool=False) -> np.ndarray:
  '''
  Computes the successor representation matrix for the given one-step transition function and gamma

  Param:
    T: one-step transition matrix
    gamma: discount factor
    extra_trans_on_sr: if true, include an extra transition at each step to avoid stalling at the current location
  
  Return:
    M: successor representation of the specified gamma
  '''
  if extra_trans_on_sr:  # include an extra transition at each step
    M = np.matmul(np.linalg.inv(np.identity(T.shape[0]) - gamma*T), T)
  else:
    M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  return M


def run_sim(n_exp: int, n_samp: int, n_row: int, n_col: int, n_rew :int, s0: Tuple[int, int], M: np.ndarray,
            rew_size: float=1.0, userandom: bool=True, idx: List[int]=[2,8,14,27,34,49,52,65,79,87], rand_rew: bool=False,
            MFC: Optional[np.ndarray]=None, add_absorb_state: bool=True,
            rho: float=1.0, beta: float=0.0, check_context_unit_norm: bool=True, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  Params:
    n_exp: number of experiments/mazes
    n_samp: number of samples drawn per experiment
    n_row: number of rows in each maze
    n_col: number of cols in each maze
    s0: starting position (same across experiments)
    M: SR matrix
    rew_size: reward size
    userandom: use random reward placement
    idx: a list of pre-determined reward positions
    rand_rew: random reward size for each maze position
    MFC: item-to-context association matrix, 
          specifying which context vector is used to update the current context vector once an item is recalled
    add_absorb_state: add an absorbing state at the end of the maze (make sure dim of M/MFC includes this if used)
    rho: TCM param, specifies how much of the previous context vector to retain
    beta: TCM param, specifies how much of the new incoming context vector to incorporate
    check_context_unit_norm: if true, checks the norm of the updated context vector is 1
    verbose: if true, prints progress (% complete)
  
  Return:
    vtrue: true value of s0 in each experiment
    vsamp: sample-based estimated value of s0 in each experiment
    samped_row: sampled rows
  '''
  vtrue = np.zeros(n_exp)
  vsamp = np.zeros((n_exp,n_samp))
  samped_row = np.zeros((n_exp,n_samp), dtype=int)
  samped_col = np.zeros((n_exp,n_samp), dtype=int)

  if not MFC:
    MFC = np.identity(M.shape[0])

  for e in range(n_exp):
    if verbose and (e+1) % (n_exp//10) == 0: print(str((e+1) * 100 // n_exp) + '%')
    maze = init_maze(n_row, n_col, n_rew, rew_size=rew_size, userandom=userandom, idx=idx)
    rvec = maze.flatten() # one-step rewards
    if rand_rew: rvec = 1 + randn(rvec.size)
    if add_absorb_state: rvec = np.append(rvec, 0)                  # no reward in the absorbing state

    state_idx = np.arange(0, n_row * n_col).reshape(maze.shape)    # state indices
    stim = np.identity(n_row * n_col + add_absorb_state)

    vtrue[e] = np.dot(M[state_idx[s0],:], rvec)                    # Compute the actual value function (as v=Mr)

    c = stim[:,state_idx[s0]]                                      # starting context vector (set to the starting state)

    samp_idx = np.negative(np.ones(n_samp, dtype=int))
    for i in range(n_samp):
      # define sampling distribution
      a = np.matmul(M.T,c)
      P = a / np.sum(a)
      assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

      # draw sample
      tmp = np.where(rand() <= np.cumsum(P))[0]
      samp_idx[i] = tmp[0]
      if samp_idx[i] >= n_row * n_col:     # sampled absorbing state:
        f = stim[:,state_idx[s0[0],s0[1]]]   # update context with starting state rather than absorbing state 
      else:
        samped_row[e,i], samped_col[e,i] = np.unravel_index(samp_idx[i], (n_row, n_col))
        f = stim[:,samp_idx[i]]

      cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
      cIN = cIN1/np.linalg.norm(cIN1)
      assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update context
      c = rho * c + beta * cIN         # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
      c = c/np.linalg.norm(c)
      if check_context_unit_norm:
        assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
    
    # compute sampled rewards
    rewsamp = rvec[samp_idx[samp_idx >= 0]]
    rewsamp = np.append(rewsamp, np.zeros(n_samp - len(rewsamp)))
    # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)
    vsamp[e,:] = rewsamp * np.sum(M[state_idx[s0],:]) 

  return vtrue, vsamp, samped_row
