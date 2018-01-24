from sklearn.ensemble import RandomForestRegressor
import timeit
import numpy as np
import sys

counts = np.loadtxt(sys.argv[1],dtype=float)


for i in range(1000):
    sample_subsample_indecies = np.random.permutation(counts.shape[0])[:int(sys.argv[2])]

    input_feature_indecies = np.random.permutation(counts.shape[1])[:int(sys.argv[3])]
    output_feature_indecies = np.random.permutation(counts.shape[1])[:int(sys.argv[3])]

    input_subsample = counts[sample_subsample_indecies]
    input_subsample = input_subsample.T[input_feature_indecies].T

    output_subsample = counts[sample_subsample_indecies]
    output_subsample = output_subsample.T[output_feature_indecies].T

    print input_subsample.shape
    print output_subsample.shape

    forest = RandomForestRegressor(n_estimators=1, min_samples_split=100, n_jobs=sys.argv[4])

    forest.fit(input_subsample.T,output_subsample.T)

    print forest.n_features_
    print forest.n_outputs_
