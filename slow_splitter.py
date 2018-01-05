import numpy as np
import sys
import math

def median_absolute_deviation(in_array):
    medians = np.median(in_array,axis=0)
    absolute_deviation = abs(in_array - np.tile(medians,(len(in_array),1)))
    return np.median(absolute_deviation, axis=0)

value_list = []

for line in open(sys.argv[1]):
    value_list.append(map(lambda x: float(x), line.split()))

print value_list
data = np.asarray(value_list)

data = data[50:]

print data

for i in range(len(data)):

    upper_deviation = median_absolute_deviation(data[i:])

    lower_deviation = median_absolute_deviation(data[:i])

    print upper_deviation

    print lower_deviation

    print str(i) + "\t" + str(math.sqrt(reduce(lambda x,y: x+y**2,upper_deviation))) + "\t" + str(math.sqrt(reduce(lambda x,y: x+y**2,lower_deviation))) + "\t" + str(math.sqrt(reduce(lambda x,y: x+y**2,upper_deviation)) + math.sqrt(reduce(lambda x,y: x+y**2,lower_deviation)))
