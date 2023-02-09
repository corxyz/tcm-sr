import pachinko_sim, bias_var, pstopSim, mazeSim
import numpy as np
import sys, math
np.set_printoptions(threshold=sys.maxsize)

figuresToMake = [3]
panelsToMake = ''

nRows, nCols = 10, 9
nRews = 15
s0 = (0,4)
addAbsorbingSt = True

T = bias_var.initTrans(nRows, nCols, addAbsorbingSt=addAbsorbingSt)
MTF_id = np.identity(nRows * nCols + addAbsorbingSt)

nExps = 10 #500
nTrials = 1000
nSamples = 1000
maxSamples = 100
pstops = [0.05, 0.5, 1.0]
gammas = [0, 0.5]

# samps = np.random.randint(1, 10, size=(100,1000))
# bias_var.plotSampleCRP(samps, 9, maxDist=8, omitZero=False)

if 1 in figuresToMake:
  # Figure 1(A): Schematic of Pachinko
  if '1a' in panelsToMake:
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    pachinko_sim.plotMaze(maze, s0)
  # Figure 1(B): transition matrices & SR
  if '1b' in panelsToMake:
    Mext, _, _ = bias_var.getSR(T, gamma=0.9, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    nStates = nRows * nCols
    stateIdx = np.arange(0,nStates).reshape((nRows, nCols))    # state indices
    pachinko_sim.plotTransSR(nRows, nCols, stateIdx, s0, T, Mext, addAbsorbingSt=addAbsorbingSt)
  
  # Figure 1(C): simuated recency and contiguity curves
  if '1c' in panelsToMake:
    rho_1c, beta_1c = 0.9, 0.1
    M_1c, M_1c_gamma1, _ = bias_var.getSR(T, gamma=0.5, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    bias_var.plotRecency(10000, nRows, nCols, nRews, s0, M_1c[:nRows*nCols, :nRows*nCols], 
                          MFC=MTF_id[:nRows*nCols, :nRows*nCols], rho=0.3, beta=0.7, addAbsorbingSt=False)
    _, _, _, _, sampRow = bias_var.runSim(1000, 10, nRows, nCols, nRews, s0, M_1c, M_1c_gamma1,
                                          MFC=MTF_id, rho=rho_1c, beta=beta_1c)
    bias_var.plotSampleCRP(sampRow, nRows, maxDist=9, omitZero=True)
  
  # Figure 1(C): TCM-SR visualization (not included) 

if 2 in figuresToMake:
  rho, beta = 1, 0
  # Figure 2(A)
  if '2a' in panelsToMake:
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)    # state indices
    stim = np.identity(nStates + addAbsorbingSt)
    gamma = 0
    Mext, _, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta)
    gamma = 0.5
    Mext, _, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta)
    
  # Figure 2(B)
  vtrues, vsamps, sampRowList = [], [], []
  for gamma in gammas:
    Mext, Mext_gamma1, Text = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    vtrue, _, vsamp, _, sampRow = bias_var.runSim(nExps, nSamples, nRows, nCols, nRews, 
                                                  s0, Mext, Mext_gamma1, MFC=MTF_id,
                                                  rho=rho, beta=beta)
    vtrues.append(vtrue)
    vsamps.append(vsamp)
    sampRowList.append(sampRow)

  if '2b' in panelsToMake:
    pachinko_sim.plotSampleDistGammas(s0, nRows, nCols, nExps, sampRowList, gammas=gammas, rho=rho, beta=beta)

  # Figure 2(C)
  if '2c' in panelsToMake:
    bias_var.plotValueEst(nSamples, vtrues, vsamps, beta, gammas=gammas, xabs_max=2)


if 3 in figuresToMake:
  rho, beta = 0, 1
  vsampList, vtrueList, eGammas = [], [], []

  gamma = gammas[0]
  Mext, Mext_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)

  if '3a' in panelsToMake:
    # Figure 3(A)
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)
    stim = np.identity(nStates + addAbsorbingSt)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta, inclLastSamp=True)

  sampRow = np.negative(np.ones((len(pstops), nExps, nTrials, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, nTrials, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  predictedGammas = np.array([pstop * gamma + (1-pstop) for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nTrials, nRows, nCols, nRews, s0, Mext, Meffect, pstop,
                                                MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta)
    sampRow[i,:,:,:] = samped
    vsamps[i,:,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve
    pstopSim.plotSampleCRP(samped, nRows, omitZero=True)
  vsampList.append(vsamps)
  vtrueList.append(veffect)
  eGammas.append(predictedGammas)

  if '3b' in panelsToMake:
    # Figure 3(B) - effective gamma version
    pstopSim.plotSampledRowseGamma(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, predictedGammas, beta)

  gamma = gammas[1]
  Mext, Mext_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)

  if '3c' in panelsToMake:
    # Figure 3(C)
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)
    stim = np.identity(nStates + addAbsorbingSt)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta, inclLastSamp=True)

  sampRow = np.negative(np.ones((len(pstops), nExps, nTrials, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, nTrials, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  predictedGammas = np.array([pstop * gamma + (1-pstop) for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nTrials, nRows, nCols, nRews, s0, Mext, Meffect, pstop,
                                                MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta)
    sampRow[i,:,:,:] = samped
    vsamps[i,:,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve
    pstopSim.plotSampleCRP(samped, nRows, omitZero=True)
  vsampList.append(vsamps)
  vtrueList.append(veffect)
  eGammas.append(predictedGammas)

  if '3d' in panelsToMake:
    # Figure 3(D) - effective gamma version
    pstopSim.plotSampledRowseGamma(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, predictedGammas, beta)

  # Figure 3(E)
  if '3e' in panelsToMake:
    # version 1: samples
    pstopSim.plotValueEst(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
    # version 2: trials
    # pstopSim.plotValueEst2(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
  
  pstopSim.plotValCorr(vtrueList, vsampList, beta, gammas, eGammas, plotFitted=True)

if 4 in figuresToMake:
  rho, beta = 0.75, 0.25
  vsampList, vtrueList, eGammas = [], [], []

  gamma = gammas[0]
  Mext, Mext_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)

  if '4a' in panelsToMake:
    # Figure 4(A)
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)
    stim = np.identity(nStates + addAbsorbingSt)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta, inclLastSamp=True)

  sampRow = np.negative(np.ones((len(pstops), nExps, nTrials, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, nTrials, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  predictedGammas = np.array([pstop * gamma + (1-pstop) for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nTrials, nRows, nCols, nRews, s0, Mext, Meffect, pstop,
                                                MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta)
    sampRow[i,:,:,:] = samped
    vsamps[i,:,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve
  vsampList.append(vsamps)
  vtrueList.append(veffect)
  eGammas.append(predictedGammas)

  if '4b' in panelsToMake:
    # Figure 4(B) - effective gamma version
    pstopSim.plotSampledRowseGamma(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, predictedGammas, beta, legend_in=True)

  gamma = gammas[1]
  Mext, Mext_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)

  if '4c' in panelsToMake:
    # Figure 4(C)
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)
    stim = np.identity(nStates + addAbsorbingSt)
    sampleIdx = pachinko_sim.drawSamples(s0, maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta, inclLastSamp=True)

  sampRow = np.negative(np.ones((len(pstops), nExps, nTrials, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, nTrials, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  predictedGammas = np.array([pstop * gamma + (1-pstop) for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Meffect, _, _ = bias_var.getSR(T, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nTrials, nRows, nCols, nRews, s0, Mext, Meffect, pstop,
                                                MFC=MTF_id, maxSamples=maxSamples, rho=rho, beta=beta)
    sampRow[i,:,:,:] = samped
    vsamps[i,:,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve
  vsampList.append(vsamps)
  vtrueList.append(veffect)
  eGammas.append(predictedGammas)

  if '4d' in panelsToMake:
    # Figure 4(D) - effective gamma version
    pstopSim.plotSampledRowseGamma(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, predictedGammas, beta, legend_in=True)

  # Figure 4(E)
  if '4e' in panelsToMake:
    # version 1: samples
    pstopSim.plotValueEst(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
    # version 2: trials
    # pstopSim.plotValueEst2(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
  
  pstopSim.plotValCorr(vtrueList, vsampList, beta, gammas, eGammas, plotFitted=True)
  
if 6 in figuresToMake:
  rho, beta = 0, 1
  pstops = [0.2, 0.5, 1]
  T = bias_var.initTrans(nRows, nCols, addAbsorbingSt=addAbsorbingSt)
  Trev = bias_var.initTrans(nRows, nCols, addAbsorbingSt=addAbsorbingSt, reversible=True)
  vsampList, vtrueList, eGammas = [], [], []
  vrefList = []

  gamma = gammas[1]
  M, M_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=False)
  Mext, Mext_gamma1, _ = bias_var.getSR(T, gamma=gamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)

  if '6a' in panelsToMake:
    # Figure 6(A)
    maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True)
    nStates = maze.size
    stateIdx = np.arange(0,nStates).reshape(maze.shape)
    stim = np.identity(nStates + addAbsorbingSt)
    sampleIdx = pachinko_sim.drawSamples((4,4), maze, stateIdx, stim, Mext, 4, rho=rho, beta=beta, MFC=M, inclLastSamp=True)

  sampRow = np.negative(np.ones((len(pstops), nExps, nTrials, maxSamples), dtype=int))
  vsamps = np.zeros((len(pstops), nExps, nTrials, maxSamples))
  vtrues = np.zeros((len(pstops), nExps))
  veffect = np.zeros((len(pstops), nExps))
  predictedGammas = np.array([pstop * gamma + (1-pstop) for pstop in pstops])
  for i, pstop in enumerate(pstops):
    eGamma = predictedGammas[i]
    Mrev, _, _ = bias_var.getSR(Trev, gamma=eGamma, hasAbsorbingSt=addAbsorbingSt, extraTransOnSR=True)
    samped, vsamp, vtrue, ve = pstopSim.runSim(nExps, nTrials, nRows, nCols, nRews, s0, Mext, Mrev, pstop, reachable=s0,
                                                randomRewards=True, MFC=Mext, maxSamples=maxSamples, rho=rho, beta=beta)
    sampRow[i,:,:,:] = samped
    vsamps[i,:,:,:] = vsamp
    vtrues[i,:] = vtrue
    veffect[i,:] = ve
    # pstopSim.plotSampleCRP(samped, nRows, omitZero=True)
  vsampList.append(vsamps)
  vtrueList.append(veffect)
  vrefList.append(vtrues)
  eGammas.append(predictedGammas)

  # Figure 6(D)
  if '6d' in panelsToMake:
    pstopSim.plotEstErrDist(vtrueList[0], vsampList[0], eGammas[0], plotFitted=True, title=r'$M^{SC}=M$')
    pstopSim.plotEstErrDist(vtrueList[0], vrefList[0], eGammas[0], plotFitted=True, isSample=False, title=r'$M^{SC}=I$')
    # pstopSim.plotValCorr(vtrueList[0], vsampList[0], eGammas[0], plotFitted=True)
    # pstopSim.plotValCorr(vtrueList[0], vrefList[0], eGammas[0], plotFitted=True, isSample=False)

  # Figure 6(E)
  if '6e' in panelsToMake:
    # version 1: samples
    pstopSim.plotValueEst(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
    # version 2: trials
    pstopSim.plotValueEst2(maxSamples, nTrials, vtrueList, vsampList, beta, gammas, eGammas)
