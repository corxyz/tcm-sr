import numpy as np
from numpy.random import rand, randn
import pylab as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from numpy import percentile, nanmean, std, nansum
from numpy.matlib import repmat
from matplotlib.gridspec import GridSpec
import math
import itertools

import bias_var, mazeSim, pachinko_sim

cm = 1/2.54  # centimeters in inches

def runSim(nExps, nTrials, nRows, nCols, nRews, s0, M, Meffect, pstop, 
            MFC=None, maxSamples=50, addAbsorbingSt=True, userandom=True, maze=None, mazeType='pachinko', mazeParams=None,
            randomRewards=False, reachable=None,
            rho=None, beta=None, ensureCnormIs1=True, plotSamples=False, verbose=False):
  # MFC (item-to-context associative matrix, specifying which context vector is used to update the current context vector once an item is recalled)
  vtrue = np.zeros(nExps)
  veffect = np.zeros(nExps)
  vsamp = np.zeros((nExps, nTrials, maxSamples))
  sampRow = np.negative(np.ones((nExps, nTrials, maxSamples), dtype=int))

  # compute rho and beta (if one of them was not provided aka None)
  if rho == None:
    rho = 1-beta
  elif beta == None:
    beta = 1-rho

  for e in range(nExps):
    if verbose and nExps >= 10 and (e+1) % (nExps//10) == 0: print(str((e+1) * 100 // nExps) + '%')
    if userandom and mazeType == 'pachinko':
      maze = pachinko_sim.initMaze(nRows, nCols, nRews, userandom=True, reachable=reachable)
    elif userandom and mazeType == '8':
      maze = mazeSim.initMaze(rewFrac=mazeParams['rewFrac'], rewSize=mazeParams['rewSize'], shape='8')
    rvec = maze.flatten() # 1-step rewards
    if randomRewards: rvec = 1 + randn(rvec.size)
    if addAbsorbingSt: rvec = np.append(rvec, 0)

    stateIdx = np.arange(0, maze.size).reshape(maze.shape)    # state indices
    stim = np.identity(maze.size + addAbsorbingSt)

    vtrue[e] = np.dot(M[stateIdx[s0],:], rvec)                # Compute the actual value function (as v=Mr)
    veffect[e] = np.dot(Meffect[stateIdx[s0],:], rvec)        # Compute the effective value function (as v=Mr)

    for t in range(nTrials):
      c = stim[:,stateIdx[s0]] # starting context vector (set to the starting state)
      sampleIdx = np.negative(np.ones(maxSamples, dtype=int))
      # roll out with a probability of stopping (pstop)
      stopped = False
      i = 0
      while not stopped:
        # define sampling distribution
        a = np.matmul(M.T, c)
        P = a / np.sum(a)
        assert np.abs(np.sum(P)-1) < 1e-10, 'P is not a valid probability distribution'

        # draw one sample
        tmp = np.where(rand() <= np.cumsum(P))[0]
        sampleIdx[i] = tmp[0]
        if sampleIdx[i] >= nRows * nCols: # entered absorbing state
          stopped = True
          break
        else:
          sampRow[e,t,i], _ = np.unravel_index(sampleIdx[i], (nRows, nCols))
          f = stim[:,sampleIdx[i]]

        cIN1 = np.matmul(MFC,f)          # PS: If gammaFC=0, cIN=s (i.e., the context vector is updated directly with the stimulus)
        cIN = cIN1/np.linalg.norm(cIN1)
        assert np.abs(np.linalg.norm(cIN)-1) < 1e-10, 'Norm of cIN is not one'
        
        # update context
        c = rho * c + beta * cIN                  # e.g.: if beta=0.5, the new stimulus contributes 50% of the new context vector
        c = c/np.linalg.norm(c)
        if ensureCnormIs1:
          assert np.abs(np.linalg.norm(c)-1) < 1e-10, 'Norm of c is not one'
        
        # do I stop now?
        stopped = stopped | (rand() <= pstop)
        i += 1
      
      # compute total sampled rewards
      rewsamp = rvec[sampleIdx[sampleIdx >= 0]]
      rewsamp = np.append(rewsamp, np.zeros(maxSamples - len(rewsamp)))
      vsamp[e,t,:] = rewsamp * np.sum(M[stateIdx[s0],:]) # scale reward samples by the sum of the row of the SR (because v=Mr, and we computed the samples based on a scaled SR)    

  return sampRow, vsamp, vtrue, veffect

def fitEffectiveGamma(sampRow, nRows, nExps, pstops, beta, gamma, s0):
  # fit effective gamma for each stopping probability
  startRow = s0[0]
  x = np.arange(0, nRows)
  olsRes = []
  for i, pstop in enumerate(pstops):
    d = sampRow[i, :, :].flatten()
    d = d[d >= 0]
    # compute average counts across experiments
    avgCnt = np.zeros(nRows)
    for r in x:
      avgCnt[r] = (d == r).sum()/float(nExps)
    avgCnt = avgCnt / avgCnt.sum()
    # ln(#samps) = X * ln(effectiveGamma) + ln(1 - effectiveGamma)
    try: firstZero = np.where(avgCnt[1:] == 0)[0][0] + startRow + 1
    except: firstZero = len(avgCnt)
    y, X = np.log(avgCnt[startRow+1:firstZero]), sm.add_constant(np.arange(0, firstZero-startRow-1), prepend=False)
    model = sm.OLS(y, X)
    res = model.fit()
    fitted = np.exp(res.params)
    olsRes.append(res)
  return olsRes

def plotSampledRowsStep(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, gamma, eGammas, beta, steps=3, normalize=False, includeReference=False):
  fig, axes = plt.subplots(1, 1 + len(pstops), gridspec_kw={'width_ratios': [2] + [1]*len(pstops)})

  # plot the board with starting position marked
  ax0 = axes[0]
  ax0.plot(s0[1]+0.5, nRows-0.5, 'c*', markersize=20)
  ax0.set_xticks(np.arange(0, nCols+1, 1))
  ax0.set_yticks(np.arange(0, nRows+1, 1))
  ax0.tick_params(length=0, labelbottom=False, labelleft=False)   
  ax0.grid()
  ax0.set_aspect('equal', adjustable='box')

  c = ['r','b','g']
  m = ['x','+','v']
  # plot the sampling distributions of different effective gammas
  for i, pstop in enumerate(pstops):
    ax1 = axes[1+i]
    y = np.arange(0, nRows, 1)
    ax1.invert_yaxis()
    alphas = np.linspace(1,0.3,num=steps)
    for step in range(steps):
      d = sampRow[i, :, :, step].flatten()
      d = d[d >= 0]
      cnt = np.array([(d == r).sum()/float(nExps*nTrials) for r in y])
      cnt[0] = np.nan
      psamp = cnt / np.nansum(cnt)
      theo = np.array([(1-gamma) * gamma ** (r-step-1) if r-step > 0 else np.nan for r in y])
      theo = theo / np.nansum(theo)
      ax1.plot(theo, y,  c[i%len(c)] + '-', linewidth=0.5, 
                alpha=alphas[step], 
                label='t={} (expected)'.format(step+1))
      ax1.plot(psamp, y, c[i%len(c)] + m[step%len(m)], 
                alpha=alphas[step], 
                label='t={} (empirical)'.format(step+1))
    ax1.set_xticks(np.arange(0, 1, .25))
    ax1.set_yticks(np.arange(0, nRows, 1))
    ax1.set_xlabel(r'$P(i_t)$')
    if i == 0: ax1.set_ylabel('Row number ' + r'$(i_t)$') # show y axis label once
    title = r'$p_{stop}=$' + str(pstop) if eGammas[i] < 0.99 else r'$p_{stop}\to 0$'
    ax1.set_title(title)
    ax1.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', prop={'size': 8})

  plt.tight_layout()
  plt.subplots_adjust(wspace=0.1)
  plt.show()

def plotSampledRowseGamma(sampRow, nRows, nCols, nExps, nTrials, s0, pstops, eGammas, beta, legend_in=False):
  figsize = [3.5*cm,3*cm] if legend_in else [5*cm,3*cm]
  fig = plt.figure(figsize=figsize, dpi=300)

  # plot the sampling distributions of different effective gammas
  y = np.arange(0, nRows, 1)
  c = ['r','b','g']
  plt.gca().invert_yaxis()
  for i, pstop in enumerate(pstops):
    gamma = eGammas[i]
    d = sampRow[i,:,:,:].flatten()
    d = d[d >= 0]
    cnt = np.array([(d == r).sum()/float(nExps) for r in y])
    cnt[0] = np.nan
    psamp = cnt / np.nansum(cnt)
    theo = np.array([(1-gamma) * gamma ** (r-1) if r > 0 else np.nan for r in y])
    theo = theo / np.nansum(theo)
    label = r'$\widetilde{\gamma}=$' + str(gamma) if gamma < 0.99 else r'$\widetilde{\gamma}\to 1$'
    plt.plot(theo, y, c[i%len(c)] + '-', linewidth=0.5, label=label)
    plt.plot(psamp, y, c[i%len(c)] + 'o', markersize=4, mew=0.8)
  plt.xticks(np.linspace(0, 1, num=5), fontsize=5)
  plt.yticks(np.arange(0, nRows, 1), fontsize=5)
  plt.xlabel(r'$P(i_t)$', size=5, labelpad=1)
  plt.ylabel('Row number ', size=5, labelpad=1)

  if legend_in: plt.legend(loc='lower right', fontsize=5)
  else: plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left', fontsize=5)
  plt.tight_layout(pad=0.6)
  plt.show()

def plotSampledRows2(sampRow, nRows, nExps, pstops, beta, eGammas, maxX=100, normalize=False):
  # sampRow has dimension len(pstops) x nExps x maxSamples
  plt.figure(1, figsize=(12,3))
  plt.clf()
  x = np.arange(0, nRows)
  c = ['r','b','g']
  # plot histogram outlines for each stopping probability
  for i, pstop in enumerate(pstops):
    gamma = eGammas[i]
    # plot reference lines for each effective gamma
    ref = np.array([np.nan] + [(1-gamma) * gamma ** (i-1) for i in range(1, nRows)])
    ref = ref / np.nansum(ref)
    label = r'$\widetilde{\gamma}=$' + str(gamma) if gamma < 0.99 else r'$\widetilde{\gamma}\to 1$'
    plt.plot(x, ref, c[i%len(c)] + '-', label=label + ' (expected)')
    d = sampRow[i, :, :].flatten()
    d = d[d >= 0]
    # compute average counts across experiments
    avgCnt = np.zeros(nRows)
    for r in x:
      avgCnt[r] = (d == r).sum()/float(nExps)
    if normalize: avgCnt = avgCnt / avgCnt.sum()
    if avgCnt[0] == 0: avgCnt[0] = np.nan
    plt.plot(x, avgCnt, c[i%len(c)] + 'x', label=label + ' (empirical)')

  # plt.xticks(np.arange(1, nRows+1, max(nRows//10,1)))
  plt.xlim([0, maxX])
  plt.xlabel('Row number ' + r'$(i_t)$')
  if normalize: plt.ylabel(r'$P(i_t)$')
  else: plt.ylabel('average samples per experiment')
  # plt.title('sampling distribution')
  plt.legend()
  plt.show()

def plotSampleCRP(sampRow, nRows, maxDist=5, beta=0, ymax=None, omitZero=True):
  nExps, nTrials, nSamples = sampRow.shape
  x = np.arange(-maxDist, maxDist+1)  # samples could come from row 0 to nRows-1 and we restrict it to the range of maxDist
  allCount = np.zeros((nExps, nRows * 2 - 1))
  count = np.zeros((nExps, nTrials, maxDist * 2 + 1))  # compute CRP curve for each experiment and each stopping probability
  # accumulate counts of relative positions of consecutive samples
  for e in range(nExps):
    for t in range(nTrials):
      samps = sampRow[e,t,:]
      for i in range(nSamples):
        if samps[i] < 0: break
        if i == 0: 
          relPos = samps[i] if samps[i] > 0 else maxDist + 1
        else: 
          relPos = samps[i] - samps[i-1]
        if abs(relPos) < nRows: allCount[e, relPos + nRows - 1] += 1
        if omitZero and relPos == 0 or abs(relPos) > maxDist: continue
        count[e, t, relPos + maxDist] += 1
  
  # compute density, mean, and standard error across experiments for each stopping probability
  sum_of_rows = count.sum(axis=2)
  y = count / sum_of_rows[:, :, np.newaxis]
  if omitZero: y[:, :, maxDist] = np.nan
  mean = np.nanmean(y, axis=(0,1))
  # plot graph
  fig, ax = plt.subplots(figsize=[1.5*cm, 3.5*cm], dpi=300)
  ax.plot(mean, x, 'ko-', markersize=2.5, linewidth=0.7)
  ax.tick_params(axis='both', which='major', length=2, direction='in',labelsize=5)
  ax.set_yticks([-maxDist,-1,1,maxDist])
  if not ymax: mean_max = np.round(np.nanmax(mean),1)
  ax.set_xticks(np.linspace(0, mean_max, 3))
  ax.set_ylabel(r'$\Delta$' + 'row', fontdict={'fontsize':5}, labelpad=2, rotation=90)
  ax.set_xlabel('Probability', fontdict={'fontsize':5}, labelpad=2)
  ax.invert_yaxis()
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  fig.subplots_adjust(left=0.3, bottom=0.25)
  # bias_var.make_square_axes(ax)
  # plt.tight_layout()
  plt.show()

def getValueBiasVar(truth, est, nExps, pstops):
  # compute bias
  bias = est.sum(axis=2) - truth
  # compute mean and percentiles of the estimator
  percentiles = [2.5, 25, 50, 75, 97.5]
  npstops = len(pstops)
  biasVar = dict()
  biasVar['bias'] = bias
  biasVar['pctl'] = np.empty((npstops, len(percentiles), nExps))
  biasVar['mean']= np.empty((npstops, nExps))
  biasVar['std'] = np.empty((npstops, nExps))
  for p in range(npstops):
    for e in range(nExps):
      biasVar['pctl'][p,:,e] = percentile(bias[p,:e+1], percentiles)
      biasVar['mean'][p,e] = nanmean(bias[p,:e+1])
      biasVar['std'][p,e] = std(bias[p,:e+1])
  return biasVar

def getValueBiasVar2(truth, est, sampRows, nExps, pstops):
  # compute bias
  tmp = est.sum(axis=2) * np.array([np.ones(nExps) * pstop for pstop in pstops])
  bias = tmp - truth
  # compute mean and percentiles of the estimator
  percentiles = [2.5, 25, 50, 75, 97.5]
  npstops = len(pstops)
  biasVar = dict()
  biasVar['bias'] = bias
  biasVar['pctl'] = np.empty((npstops, len(percentiles), nExps))
  biasVar['mean']= np.empty((npstops, nExps))
  biasVar['std'] = np.empty((npstops, nExps))
  for p in range(npstops):
    for e in range(nExps):
      biasVar['pctl'][p,:,e] = percentile(bias[p,:e+1], percentiles)
      biasVar['mean'][p,e] = nanmean(bias[p,:e+1])
      biasVar['std'][p,e] = std(bias[p,:e+1])
  return biasVar

def plotValueBiasVar(nExps, biasVar, eGammas, beta=1, gamma=1):
  mean = biasVar['mean']
  pctl = biasVar['pctl']
  bias = biasVar['bias']
  std  = biasVar['std']

  nRows, nCols = math.ceil(len(eGammas)/4), 4
  plt.figure(1, figsize=[nCols * 2.5, nRows * 2.5])
  plt.clf()

  for i, eGamma in enumerate(eGammas):
    plt.subplot(nRows, nCols, i+1, title=r'$\widetilde{\gamma}=$' + str(round(eGamma, 5)))
    lines1 = plt.semilogx(repmat(np.arange(0,nExps),5,1).T, pctl[i,:,:].T)
    lines2 = plt.semilogx(np.arange(0,nExps), mean[i,:])
    plt.grid(True)
    # plt.ylim([-3,3])
    plt.setp(lines1[0], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='95% CI')
    plt.setp(lines1[4], color=[0.5,0.5,0.5], linewidth=1, linestyle=':', label='50% CI')
    plt.setp(lines1[1], color=[0.3,0.3,0.3], linewidth=1, linestyle='--', label='Median')
    plt.setp(lines1[3], color=[0.3,0.3,0.3], linewidth=1, linestyle='--')
    plt.setp(lines1[2], color=[0.1,0.1,0.1], linewidth=1, linestyle='-', label='Mean')
    plt.setp(lines2, color=[0,0,0.8], linewidth=2, linestyle='-')
    plt.xlabel('# experiments averaged')
    plt.ylabel(r'$v_{samp} - v_{\widetilde{\gamma}}$')
  
  plt.tight_layout()
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left')
  plt.show()

def getGammaEst(olsRes, nExps, pstops):
  est = np.exp(np.array([res.params[0] for res in olsRes]))
  ci = np.exp(np.array([res.conf_int() for res in olsRes]))
  gammaCI = ci[:,0,:]
  # store LSE estimates and 95% CIs
  d = dict()
  d['est'] = est
  d['ci95'] = gammaCI
  return d

def plotGammaEst(nExps, d, predicted, pstops, beta=1, gamma=1):
  est = d['est']
  ci = d['ci95']

  plt.figure()
  plt.clf()
  x = np.arange(0, len(pstops))
  plt.plot(x, est, 'o', label='LSE estimate')
  plt.fill_between(x, ci[:,0], ci[:,1], color='b', alpha=.1)
  plt.plot(x, np.round(predicted, decimals=5), 'x', label='hypothesis')
  plt.xticks(x, pstops)
  plt.xlabel(r'$p_{stop}$')
  plt.ylabel('Effective gamma')
  plt.title(r'$\beta=$' + str(beta) + r'$, \gamma=$' + str(gamma))
  
  plt.tight_layout()
  plt.legend()
  plt.show()

def plotResidual(olsRes, pstops, beta=1, gamma=1):
  nRows, nCols = math.ceil(len(olsRes)/4), 4
  plt.figure(1, figsize=[nCols * 2.5, nRows * 2.5])
  plt.clf()
  for i, res in enumerate(olsRes):
    plt.subplot(nRows, nCols, i+1, title=r'$p_{stop}=$' + str(pstops[i]) + r'$, \gamma=$' + str(gamma))
    plt.scatter(res.fittedvalues, res.resid, s=2)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xlabel(r'$\ln(P_{sample}^r)$')
    plt.ylabel('Residual')
  plt.tight_layout()
  plt.show()

def getValueEstPctl(vsamp, vtrue, beta, avgTrials=False):
  nExps, nTrials, nSamples = vsamp.shape
  percentiles = [2.5, 25, 50, 75, 97.5]
  if beta == 0:
    bias = vsamp - repmat(vtrue, nSamples, 1).T
    mean = np.empty(nSamples)
    pctl = np.empty((len(percentiles), nSamples))
    for s in range(nSamples):
      mean[s] = nanmean(nanmean(bias[:,:s+1],axis=1))
      pctl[:,s] = percentile(nanmean(bias[:,:s+1],axis=1), percentiles)
  elif avgTrials: # average across trials
    mean = np.empty(nTrials)
    pctl = np.empty((len(percentiles), nTrials))
    bias = np.empty((nExps, nTrials))
    for t in range(nTrials):
      bias[:,t] = nanmean(nansum(vsamp[:,:t+1,:], axis=2), axis=1) - vtrue
      mean[t] = nanmean(bias[:,t])
      pctl[:,t] = percentile(bias[:,t], percentiles)
  else:
    mean = np.empty(nTrials*nSamples)
    pctl = np.empty((len(percentiles), nTrials*nSamples))
    bias = np.empty((nExps, nTrials*nSamples))
    for s in range(nTrials*nSamples):
      t = math.ceil((s+1) / nSamples) # compute number of trials included
      r = (s+1) % nSamples            # compute number of samples included that belong to the last (possibly incomplete) trial
      if r == 0: r = nSamples
      if t == 1: est = np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1)
      else: est = np.append(nansum(vsamp[:,:t-1,:], axis=2)*beta, np.expand_dims(nansum(vsamp[:,t-1,:r], axis=1)*beta, axis=1), axis=1)
      bias[:,s] = nanmean(est, axis=1) - vtrue
      mean[s] = nanmean(bias[:,s])
      pctl[:,s] = percentile(bias[:,s], percentiles)
  return mean, pctl, bias

def plotValueEst(nSamples, nTrials, vtruesList, vsampsList, beta, gammas, eGammas, xabs_max=5):
  # nSamples: int
  # vtruesList dim = n_gammas * n_pstops * nExps
  # vsampsList dim = n_gammas * n_pstops * nExps * nTrials * nSamples
  # beta: float
  # gammas: list of length n_gammas
  # eGammas dim = n_gammas * n_pstops
  nRows, nCols = 2*len(gammas), len(eGammas[0])
  fig = plt.figure(1, figsize=[nCols*3*2*cm, nRows*2.4*cm], dpi=300)
  plt.clf()
  gs = GridSpec(nRows, nCols*3, figure=fig, hspace=0.73)

  axes, axes2 = [], []
  ymax, ymax2 = 0, 0
  for k, gamma in enumerate(gammas):
    vsamps, vtrues = vsampsList[k], vtruesList[k]
    for i, egamma in enumerate(eGammas[k]):
      vsamp, vtrue = vsamps[i], vtrues[i]
      vsamp[vsamp == 0] = np.nan
      mean, pctl, bias = getValueEstPctl(vsamp, vtrue, beta)
      # plot bias density plots
      for j, n in enumerate([10, 100, 1000]):
        if beta != 0 and nTrials*nSamples >= n or beta == 0 and nSamples >= n:
          ax = fig.add_subplot(gs[2*k,i*3+j])
          axes.append(ax)
          ax.set_title(r'$n=$' + str(n), size=5, pad=2)
          ax.hist(bias[:,n-1], bins=np.arange(-xabs_max, xabs_max+0.1, 0.1), density=True)
          # xabs_max = max(abs(max(ax.get_xlim(), key=abs)), xabs_max)
          ymax = max(max(ax.get_ylim()), ymax)
          ax.set_xlabel('bias', fontdict={'fontsize':5}, labelpad=2)
          ax.xaxis.labelpad = 2
          if j == 0 and i == 0:
            ax.set_ylabel('density', fontdict={'fontsize':5}, labelpad=2)
            ax.tick_params(axis='both', which='major', labelsize=5, length=2, pad=1)
          else:
            ax.tick_params(labelleft=False, labelsize=5, length=2, pad=1)
      # plot estimated value plot
      ax = fig.add_subplot(gs[2*k+1,i*3:(i+1)*3])
      ax.set_title(r'$\widetilde{\gamma}=$' + str(egamma), size=7, pad=2)
      axes2.append(ax)
      colors = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,0.1,0.1],[0.3,0.3,0.3],[0.5,0.5,0.5]]
      ls = [':','--','-','--',':']
      labels = ['95% CI', '50% CI', 'Median', '', '']
      lines1 = ax.semilogx(repmat(np.arange(pctl.shape[1]),5,1).T, pctl.T, linewidth=0.7)
      for l, line in enumerate(ax.get_lines()):
        line.set_linestyle(ls[l])
        line.set_color(colors[l])
        if (len(labels[l]) > 0): line.set_label(labels[l])
      lines2 = ax.semilogx(np.arange(mean.shape[0]), mean, color=[0,0,0.8], linewidth=1, linestyle='-', label='Mean')
      ax.grid(True)
      ax.tick_params(labelsize=5, length=2, pad=0)
      ax.set_xlabel(r'$n$', fontdict={'fontsize':5}, labelpad=0)
      if i == 0: ax.set_ylabel(r'$\hat{v}-v_{true}$', fontdict={'fontsize':5})
      else: ax.tick_params(labelleft=False)
      ymax2 = max(max(ax.get_ylim()), ymax2)
    ax.legend(bbox_to_anchor=(1.04, 0), loc='lower left', fontsize=5)
  
  for ax in axes:
    ax.set_ylim([0, ymax])
  
  ymax2 = math.floor(ymax2)
  for ax in axes2:
    ax.set_ylim([-ymax2, ymax2])

  fig.subplots_adjust(right=0.8, top=0.97, bottom=0.07)
  # plt.savefig('f3e.pdf', dpi=300)
  plt.show()

def plotValueEst2(nSamples, nTrials, vtruesList, vsampsList, beta, gammas, eGammas):
  # nSamples: int
  # vtruesList dim = n_gammas * n_pstops * nExps
  # vsampsList dim = n_gammas * n_pstops * nExps * nTrials * nSamples
  # beta: float
  # gammas: list of length n_gammas
  # eGammas dim = n_gammas * n_pstops
  nRows, nCols = 2*len(gammas), len(eGammas[0])
  fig = plt.figure(1, figsize=[nCols*4, nRows*2])
  plt.clf()
  gs = GridSpec(nRows, nCols*3, figure=fig)

  for k, gamma in enumerate(gammas):
    vsamps, vtrues = vsampsList[k], vtruesList[k]
    for i, egamma in enumerate(eGammas[k]):
      vsamp, vtrue = vsamps[i], vtrues[i]
      vsamp[vsamp == 0] = np.nan
      mean, pctl, bias = getValueEstPctl(vsamp, vtrue, beta, avgTrials=True)
      # plot bias density plots
      axes = []
      xabs_max, ymax = 0, 0
      for j, n in enumerate([10, 100, 1000]):
        if beta == 1 and nTrials >= n or beta == 0 and nSamples >= n:
          ax = fig.add_subplot(gs[2*k,i*3+j])
          axes.append(ax)
          ax.set_title('{} trials'.format(n), fontdict={'fontsize':8})
          ax.hist(bias[:,n-1], bins=20, density=True)
          xabs_max = max(abs(max(ax.get_xlim(), key=abs)), xabs_max)
          ymax = max(max(ax.get_ylim()), ymax)
          ax.set_xlabel('bias', fontdict={'fontsize':8})
          ax.set_ylabel('density', fontdict={'fontsize':8})
          ax.tick_params(axis='both', which='major', labelsize=8)
      for ax in axes:
        ax.set_xlim([-xabs_max, xabs_max])
        ax.set_ylim([0, ymax])
      # plot estimated value plot
      ax = fig.add_subplot(gs[2*k+1,i*3:(i+1)*3], title=r'$\widetilde{\gamma}=$' + str(egamma))
      colors = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,0.1,0.1],[0.3,0.3,0.3],[0.5,0.5,0.5]]
      ls = [':','--','-','--',':']
      labels = ['95% CI', 'Median', 'Mean', '', '50% CI']
      lines1 = ax.semilogx(repmat(np.arange(pctl.shape[1]),5,1).T, pctl.T)
      for i, line in enumerate(ax.get_lines()):
        line.set_linestyle(ls[i])
        line.set_color(colors[i])
        if (len(labels[i]) > 0): line.set_label(labels[i])
      lines2 = ax.semilogx(np.arange(0,mean.shape[0]), mean, color=[0,0,0.8], linewidth=2, linestyle='-')
      ax.grid(True)
      ax.axhline(y=vtrue.mean(), color='g', linestyle='-', label=r'$v_{true}$')
      ax.set_xlabel('# trials', fontdict={'fontsize':10})
      ax.set_ylabel(r'$\hat{v}$', fontdict={'fontsize':10})
  
  fig.subplots_adjust(right=0.8)
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left')
  plt.tight_layout()
  plt.show()

def plotValCorr(veffectList, vsampList, beta, gammas, eGammas, plotFitted=False, isSample=True):
  # veffectList dim = n_gammas * n_pstops * nExps
  # vsampList dim = n_gammas * n_pstops * nExps * nTrials * nSamples
  # gammas: list of length n_gammas
  # eGammas dim = n_gammas * n_pstops

  nRows, nCols = len(gammas), len(eGammas[0])
  fig, axes = plt.subplots(nRows, nCols, figsize=[6.6*cm, 4.4*cm], dpi=300)

  for i, gamma in enumerate(gammas):
    vsamps, ves = vsampList[i], veffectList[i]
    for j, eGamma in enumerate(eGammas[i]):
      vsamp, ve = vsamps[j], ves[j]
      if isSample:
        sampEst = nanmean(nansum(vsamp, axis=2), axis=1)*beta
      else:
        sampEst = vsamp
      # plot scatter plot of value estimates against effective value (each point represents one experiment/board)
      axes[i,j].scatter(ve, sampEst, c='k', s=0.5)
      maxlim = max(axes[i,j].get_xlim()[1], axes[i,j].get_ylim()[1])
      minlim = min(axes[i,j].get_xlim()[0], axes[i,j].get_ylim()[0])
      axes[i,j].set_xlim([minlim, maxlim])
      axes[i,j].set_ylim([minlim, maxlim])
      # plot LSE line
      if plotFitted:
        m, b = np.polyfit(ve, sampEst, 1)
        print('gamma={}, egamma={}, fitted slope={}, intercept={}'.format(gamma, eGamma, m, b))
        axes[i,j].plot(ve, m*ve + b, c='c', lw=0.8)
      # plot reference diagonal line
      axes[i,j].plot([0, 1], [0, 1], transform=axes[i,j].transAxes, ls='--', c='gray', lw=0.4)
      axes[i,j].set_title(r'$\widetilde{\gamma}$=' + str(str(round(eGamma, 5))), size=7, pad=2)

  for ax in axes.flat:
    ax.set_xlabel(r'$v_{\widetilde{\gamma}}$', fontdict={'fontsize':5}, labelpad=1)
    ax.set_ylabel(r'$\hat{v}$', fontdict={'fontsize':5}, labelpad=1)
    ax.tick_params(labelsize=5, length=2, pad=1)
    plt.setp(ax.spines.values(), linewidth=0.5)
  
  plt.tight_layout(pad=0.25)
  plt.show()

def plotEstErrDist(veffectList, vsampList, eGammas, plotFitted=False, isSample=True, title='Density plot'):
  # plot the error distribution of value estimates against the true value
  # veffectList dim = n_gammas * n_pstops * nExps
  # vsampList dim = n_gammas * n_pstops * nExps * nTrials * nSamples
  # gammas: list of length n_gammas
  # eGammas dim = n_gammas * n_pstops
  c = ['r','b','g']
  kwargs = dict(kde_kws={'linewidth':1})
  fig, ax = plt.subplots(figsize=[6*cm, 3*cm], dpi=300)

  for i, eGamma in enumerate(eGammas):
    vsamp, truth = vsampList[i], veffectList[i]
    if isSample:
      sampEst = nanmean(nansum(vsamp, axis=2), axis=1)
    else:
      sampEst = vsamp
    # plot distribution
    sns.kdeplot(data=(sampEst-truth).T, label=r'$\tilde{\gamma}=$' + str(eGamma), ax=ax, linewidth=1, legend=False, color=c[i%len(c)])
  
  plt.title(title, fontdict={'fontsize':7}, pad=2)
  plt.xlabel(r'$\hat{v}-v_{\widetilde{\gamma}}}$', fontsize=5, labelpad=1)
  plt.ylabel('Density', fontsize=5, labelpad=1)
  ax.tick_params(axis='both', which='major', labelsize=5)
  plt.xlim([-15, 15])
  plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left', fontsize=5)
  plt.tight_layout(pad=0.25)
  plt.rcParams['figure.dpi'] = 300
  fig.subplots_adjust(right=0.7)
  plt.show()

