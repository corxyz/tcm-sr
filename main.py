import pachinko_sim, bias_var, pstopSim
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

nRows, nCols = 10, 9
nRews = 5
s0 = (0,4)
rho, beta = 1, 0
gamma = 0.9
simScheme = 'gamma'           # available options: gamma, pstop, fixedDraw, gammas

# maze = pachinko_sim.initMaze(nRows, nCols, nRews,userandom=False)
# nStates = maze.size
# stateIdx = np.arange(0,nStates).reshape(maze.shape)    # state indices
# stim = np.identity(nStates)

# T = pachinko_sim.initTrans(nStates, numRows, numCols, addAbsorbingSt=True)
# M, T = pachinko_sim.getSR(T, gamma=0.99, hasAbsorbingSt=True)
# pachinko_sim.drawSamples((0,4), maze, stateIdx, stim, M, 25, rho=1)

T = bias_var.initTrans(nRows, nCols, addAbsorbingSt=True)
M, M_gamma1, T1 = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=True, extraTransOnSR=False)
Mext, Mext_gamma1, Text = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=True, extraTransOnSR=True)

nExps = 100
nSamples = 1000
maxSamples = 50
pstops = np.round_(np.linspace(.05, 1., num=15), decimals=5)
MTF_id = np.identity(M.shape[0])

if simScheme == 'gamma':
  vtrue, vgamma1, vsamp, stateIdx, sampRow = bias_var.runSim(nExps, nSamples, nRows, nCols, nRews, 
                                                              s0, Mext, Mext_gamma1, MFC=MTF_id,
                                                              rho=rho, beta=beta, randomRewards=True)
  # vsampBiasVar = bias_var.getBiasVarPctl(vtrue, vgamma1, vsamp, nSamples, s0, M, nRows, stateIdx)
  # bias_var.plotBiasVar(nSamples, vsampBiasVar, gamma=0.89)
  # bias_var.plotSampledRows(sampRow, nRows, nExps, nSamples, beta, gamma, normalize=False, includeReference=True)
  bias_var.plotSampleCRP(sampRow, nRows, maxDist=9, omitZero=True)

if simScheme == 'pstop':
  sampRow = np.negative(np.ones((len(pstops), nExps, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  maxGamma = gamma - (1-gamma) * beta**2 + 2 * (1-gamma) * beta
  predictedGammas = np.array([pstop * gamma + (1-pstop) * maxGamma for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=True, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nRows, nCols, nRews, s0, Mext, Meffect, pstop,
                                                MFC=MTF_id, maxSamples=maxSamples, randomRewards=False, rho=rho, beta=beta)
    sampRow[i,:,:] = samped
    vsamps[i,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve

  if beta == 0: biasVar = pstopSim.getValueBiasVar2(veffect, vsamps, sampRow, nExps, pstops)
  else: biasVar = pstopSim.getValueBiasVar(veffect, vsamps, nExps, pstops)
  pstopSim.plotValueBiasVar(nExps, biasVar, predictedGammas, beta=beta)

  olsRes = pstopSim.fitEffectiveGamma(sampRow, nRows, nExps, pstops, beta, gamma, s0)
  pstopSim.plotResidual(olsRes, pstops, beta=beta, gamma=gamma)
  estGamma = pstopSim.getGammaEst(olsRes, nExps, pstops)
  pstopSim.plotGammaEst(nExps, estGamma, predictedGammas, pstops, beta=beta, gamma=gamma)

  pstopSim.plotSampledRows(sampRow, nRows, nExps, pstops, beta, gamma, includeReference=True, normalize=True)
  pstopSim.plotSampledRows(sampRow, nRows, nExps, pstops, beta, gamma, includeReference=False, normalize=False)
  pstopSim.plotSampleCRP(sampRow, nRows, pstops, maxDist=9, omitZero=True)

if simScheme == 'fixedDraw':
  maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=False)
  nStates = maze.size
  stateIdx = np.arange(0,nStates).reshape(maze.shape)    # state indices
  stim = np.identity(nStates)
  sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta)

if simScheme == 'gammas':
  gammas = [0, 0.5, 0.9]
  vtrues, vsamps, sampRowList = [], [], []
  for gamma in gammas:
    Mext, Mext_gamma1, Text = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=True, extraTransOnSR=True)
    vtrue, vgamma1, vsamp, stateIdx, sampRow = bias_var.runSim(nExps, nSamples, nRows, nCols, nRews, 
                                                              s0, Mext, Mext_gamma1, MFC=MTF_id,
                                                              rho=rho, beta=beta, randomRewards=True)
    vtrues.append(vtrue)
    vsamps.append(vsamp)
    sampRowList.append(sampRow)
  # pachinko_sim.plotSampleDist(s0, nRows, nCols, nExps, sampRowList, gammas=gammas, rho=rho, beta=beta)
  bias_var.plotValueEst(nSamples, vtrues, vsamps, gammas=gammas)
  # bias_var.plotMultipleBiasVar(nSamples, vtrues, vsamps, gammas=gammas)
