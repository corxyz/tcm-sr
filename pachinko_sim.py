import numpy as np
from numpy.random import randint, choice
import util

########################################
#  CONSTANT
########################################

cm = 1/2.54  # centimeters in inches

########################################
#  HELPER FN
########################################

def init_trans(n_row, n_col, add_absorb_state=False, reversible=False):
  '''
  Initialize the one-step transition matrix of Plinko

  Params:
    n_row: number of rows (int)
    n_col: number of columns (int)
    add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    reversible: if true, both upward and downward transitions are possible;
                if false, only downward transitions are allowed (i.e. standard Pachinko) (optional, bool)
  
  Return:
    T: one-step transition matrix (numpy array)
  '''
  n_state = n_row * n_col
  if add_absorb_state: 
    T = np.zeros([n_state+1, n_state+1])
    T[n_state, n_state] = 1
  else: T = np.zeros([n_state, n_state])

  for s in range(n_state):
    i, j = np.unravel_index(s, (n_row, n_col))
    neighbors = []
    if i < n_row-1:   # not at the bottom
      if j > 0 and j < n_col-1: # not at the walls
        neighbors += [(i+1, j-1), (i+1, j+1)]
        if reversible and i > 0: neighbors += [(i-1, j-1), (i-1, j+1)]
      elif j == 0:                # left wall
        neighbors.append((i+1, j+1))
        if reversible and i > 0: neighbors.append((i-1, j+1))
      else:                       # right wall
        neighbors.append((i+1, j-1))
        if reversible and i > 0: neighbors.append((i-1, j-1))
      for neighbor in neighbors:
        T[s, np.ravel_multi_index(neighbor, (n_row, n_col))] = 1/len(neighbors)
    else:               # at bottom
      if add_absorb_state: T[s, n_state] = 1
      else: T[s, s] = 1
  return T

def get_reachable_states(n_row, n_col, n_row_excl, s0):
  '''
    Retrieve a list of Plinko board locations that are reachable by a ball dropped at state s0

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      n_row_excl: number of rows to exclude (including the first row to drop the ball)
                e.g., if 1, only the first row is excluded; if 2, the first and the last rows are excluded
                (int)
      s0: starting position (same across experiments) (int tuple)
  '''
  T = init_trans(n_row * n_col, n_row, n_col)
  M = util.get_sr(T, gamma=0.99999)
  start = np.ravel_multi_index(s0, (n_row, n_col))
  reachables = np.nonzero(M[start, :(n_row-n_row_excl+1)*n_col])[0]
  return list(reachables[1:])

def init_maze(n_row, n_col, n_rew, 
             rew_size=1, userandom=False, idx=[2,8,14,27,34,49,52,65,79,87], 
             reachable=None, n_row_excl=4, force_row=None):
  '''
    Initialize a Plinko board

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      n_rew: number of rewards (int)
      rew_size: reward magnitude (optional, number)
      userandom: use random reward placement (optional, bool)
      idx: pre-determined reward positions (optional, list)
      reachable: only place rewards in states reachable from the specified state(s) 
                 (optional, None, int tuple, or list of int tuples)
      n_row_excl: number of rows to exclude (including the first row to drop the ball)
                e.g., if 1, only the first row is excluded; if 2, the first and the last rows are excluded
                (int)
      force_row: make sure there is at least one reward on the given row, 0-indexed (optional, None or int)
    
    Return:
      maze: Plinko board (numpy array)
  '''
  assert(n_rew > 0)
  # reachable could be (i) None (ii) a tuple indicating the state position (iii) a list of tuples (states)
  maze = np.zeros([n_row, n_col])
  if userandom:
    if reachable:
      allReachables = []
      if type(reachable) is tuple:
        allReachables = get_reachable_states(n_row, n_col, n_row_excl, reachable)
      elif type(reachable) is list:
        for s0 in reachable:
          allReachables += get_reachable_states(n_row, n_col, n_row_excl, s0)
        allReachables = list(set(allReachables))
      p = np.random.permutation(allReachables)

      rew_idx = p[:n_rew]
      if force_row: # force at least one reward to be placed in the row specified
        for pos in p: 
          if force_row * n_col <= pos < (force_row+1) * n_col:
            rew_idx[0] = pos
            break
      rew_idx = np.unravel_index(rew_idx, (n_row, n_col))
      # Place rewards on the maze
      maze[rew_idx] =  rew_size
    else:
      p = np.random.permutation((n_row - n_row_excl) * n_col)
      rew_idx = p[:n_rew]
      rew_idx = np.unravel_index(rew_idx, (n_row, n_col))
      # Place rewards on the maze (avoiding first and n_row_excl-1 last rows)
      maze_middle = np.zeros([n_row-n_row_excl,n_col])
      maze_middle[rew_idx] = rew_size
      if n_row_excl > 1: maze[1:-n_row_excl+1,:] = maze_middle
      else: maze[1:,:] =  maze_middle
  else:
    rew_idx = np.unravel_index(idx[:min(len(idx),n_rew)], (n_row, n_col))
    maze[rew_idx] = rew_size
  return maze

def init_biased_maze(n_row, n_col, n_rew, rew_size=1, rew_bias=[0.5, 0.5], n_row_excl=4):
  '''
    Initialize a Plinko board with non-uniform reward placement
    Specifically, the left and right halves of the board are expected to 
    contain different amounts of rewards

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      n_rew: number of rewards (int)
      rew_size: reward magnitude (optional, number)
      rew_bias: probabilities of an arbitrary reward being in the left vs right side of the board (optional, list)
      n_row_excl: number of rows to exclude (including the first row to drop the ball)
                e.g., if 1, only the first row is excluded; if 2, the first and the last rows are excluded
                (int)
    
    Return:
      maze: Plinko board (numpy array)
  '''
  assert(n_rew > 0)
  maze = np.zeros([n_row, n_col])
  bias = np.array(rew_bias)
  bias /= bias.sum()

  # sample reward location with specified sampling bias
  a = np.arange((n_row - n_row_excl) * n_col)
  w = np.vectorize(lambda x: bias[0] if x % n_col < n_col / 2 else bias[1])(a)
  p = w / w.sum()
  rew_idx = np.unravel_index(choice(a, size=n_rew, replace=False, p=p), (n_row, n_col))

  # Place rewards on the maze (avoiding first and n_row_excl-1 last rows)
  maze_middle = np.zeros([n_row-n_row_excl, n_col])
  maze_middle[rew_idx] = rew_size
  if n_row_excl > 1: maze[1:-n_row_excl+1,:] = maze_middle
  else: maze[1:,:] =  maze_middle

  return maze

def rand_step_plinko(n_row, n_col, curRow, curCol, add_absorb_state=False):
  '''
    Take one step in a random direction on the Plinko board

    Params:
      n_row: number of rows (int)
      n_col: number of columns (int)
      curRow: current row number, 0-indexed (int)
      curCol: current column number, 0-indexed (int)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    
    Return:
      (next row number, next column number)
  '''
  directions = []
  if curRow == n_row - 1: 
    if add_absorb_state:  # move to absorbing state; otherwise stay at the same place
      curRow, curCol = n_row, 0
    return curRow, curCol
  elif 0 < curCol < n_col - 1: directions = [(1,1),(1,-1)]
  elif curCol == 0: directions = [(1,1)]
  else: directions = [(1,-1)]
  step = directions[randint(len(directions))]
  curRow += step[0]
  curCol += step[1]
  return curRow, curCol

def get_rand_traj(maze, s0, add_absorb_state=True):
  '''
    Simulate a random Plinko trajectory

    Params:
      maze: Plinko board (numpy array)
      s0: starting position (same across experiments) (int tuple)
      add_absorb_state: whether to add an absorbing state at the end of the board or not (optional, bool)
    
    Return:
      traj: a trajectory (list of int tuples)
  '''
  # run one random Pachinko game (number of steps is equal to the number of rows - 1)
  n_row, n_col = maze.shape
  curRow, curCol = s0
  traj = [(curRow, curCol)]           # keep track of the trajectory
  while curRow < n_row:
    curRow, curCol = rand_step_plinko(n_row, n_col, curRow, curCol, add_absorb_state=add_absorb_state)   # take one random step in the maze
    traj.append((curRow, curCol))
  return traj
