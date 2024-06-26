import numpy as np

def get_sr(T, gamma=1, extra_trans_on_sr=False):
  '''
    Compute the successor representation matrix of the given transition function and gamma

    Params:
      T: one-step transition matrix (numpy array)
      gamma: discount factor (optional, float)
      extra_trans_on_sr: whether to count visit upon entering or exiting a state (optional, bool)
                    (if using the definition in Zhou et al. (2024), this should be set to True)
    
    Return:
      M: successor representation of the specified gamma (numpy array)
  '''
  M = np.linalg.inv(np.identity(T.shape[0]) - gamma*T)
  if extra_trans_on_sr:
    M = np.matmul(M, T)
  return M
