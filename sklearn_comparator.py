from sklearn.ensemble import RandomForestRegressor
import timeit
import numpy as np
import sys

counts = np.loadtxt(sys.argv[1],dtype=float)


for i in range(10):
    sample_subsample_indecies = np.random.permutation(counts.shape[0])[:int(sys.argv[2])]
    feature_subsample_indecies = np.random.permutation(counts.shape[1])[:int(sys.argv[3])]

    subsample = counts[sample_subsample_indecies]
    subsample = subsample.T[feature_subsample_indecies].T

    print subsample.shape

    forest = RandomForestRegressor(n_estimators=1, min_samples_split=100)

    forest.fit(subsample,subsample)

    print forest.n_features_
    print forest.n_outputs_
