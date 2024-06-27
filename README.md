# TCM-SR

This repository contains all the code scripts used to generate the simulations and plots included in the following paper:

> Zhou, C. Y., Talmi, D., Daw, N. D., Mattar, M. G. (in press). Episodic retrieval for model-based evaluation in sequential decision tasks. _Psychological Review_.

The preprint is available on [PsyArXiv](https://osf.io/preprints/psyarxiv/3sqjh).

## Dependencies

* Python (version: 3.9)
* Numpy (version: 1.24.2)
* Scipy (version: 1.10.1)
* Matplotlib (version: 3.8.1)
* Seaborn (version: 0.12.2)

## Overview

Below is a breakdown of the modules contained in this repository:

- `tcm_sr_demo`: a standalone demo of the basic TCM-SR algorithm
- `make_paper_figs`: most simulations illustrated in figures 1-4 and figure 6 of the paper, including (1) i.i.d. sampling; (2) generalized rollouts; (3) intermediate sampling schemes; (4) using experimental context in retrieval
- `pachinko_sim`: Plinko game initialization and example observations (trajectories)
- `bias_var`: i.i.d. sampling simulations + analysis (Simulation 1)
- `pstop_sim`: generalized rollouts and intermediate sampling simulations + analysis (Simulation 2, 3, 5)
- `limited_samp`: model-based evaluation in the small sample regime + analysis (Simulation 4, Figure 5)
- `learn_sr`: emotional modulated episodic sampling + analysis (Simluation 4, Figure 5)
- `choice`: all decision behavior simulation + analysis (Simulation 1-5)
- `plotting`: plotting helper functions
- `util`: utility functions

## Quickstart

As an example, to simulate model-based evaluation using episodic memory dynamics and generalized rollouts, run

`python make_paper_figs.py`

The default hyperparameters should reproduce panels in Figure 3 (_Simulation 2: Recall-dependent context updates lead to rollouts_). For other simulations, please refer to the paper manuscript and the script `make_paper_figs.py`.
