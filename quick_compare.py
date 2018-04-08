#!/usr/bin/env python

import sys
import numpy as np
from scipy.stats import pearsonr 

truth = np.loadtxt(sys.argv[1])
pred = np.loadtxt(sys.argv[2])

pred[np.isnan(pred)] = 0

print "Correlation:"
print pearsonr(truth.flatten(),pred.flatten())

