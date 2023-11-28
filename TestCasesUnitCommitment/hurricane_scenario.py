import scipy.io
import numpy as np

data = scipy.io.loadmat('Sample_hurricane.mat')
sample = data['sample']

from StochasticOptimization.scenario_reduction import ScenarioReduction

ns = sample.shape[0]
weight = np.ones(ns)/ns
scenario_reduction = ScenarioReduction()

(scenario_reduced, weight_reduced) = scenario_reduction.run(scenario=sample, weight=weight, n_reduced=int(0.95*ns),power=2)

scipy.io.savemat('reduced_scenario',{'sample':scenario_reduced,'weight':weight_reduced})