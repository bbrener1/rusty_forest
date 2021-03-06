#!/usr/bin/env python

import sys
import numpy as np
from scipy.stats import pearsonr
from math import isnan

truth = np.loadtxt(sys.argv[1])
pred = np.loadtxt(sys.argv[2])

pred[np.isnan(pred)] = 0

print("Correlation:")
print(pearsonr(truth.flatten(),pred.flatten()))

print("Mean Feature Correlation:")
correlations = []
for i in range(truth.T.shape[0]):
    correlations.append(pearsonr(truth.T[i],pred.T[i])[0])
    if isnan(correlations[-1]):
        correlations[-1] = 0
print(np.mean(correlations))

print("Mean Absolute Error")
print(np.mean(np.abs(truth - pred)))
