#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.special
from scipy.stats import linregress
import re


import numpy as np

from scipy.cluster import hierarchy as hrc
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

import sys



def read_tree(location,header):

    print location

    tree_string = open(location).read()
    tree_nodes = tree_string.split("!ID:")
    # print tree_nodes
    nodes = {}
    root = []
    for node_string in tree_nodes[1:]:
        node_list = node_string.split("\n")
        # print node_list
        # print len(node_list)
        node = {}
        node["id"] = node_list[0]
        node["children"] = node_list[1].split("!C:")[1:]
        node["parent"] = node_list[2].split(":")[1:]
        if node_list[3].split(":")[1] != "None":
            node["feature"] = header[int(node_list[3].split(":")[1].lstrip("Some(\"").rstrip("\")"))]
        else:
            node["feature"] = "None"
        if node_list[4].split(":")[1] != "None":
            node["split"] = float(node_list[4].lstrip("Split:Some(").rstrip(")"))
        node["output_features"] = re.findall('"(.*?)"', node_list[6])
        # print re.findall('(\d+\.*\d*)', node_list[7])
        node["medians"] = map(lambda x: float(x),re.findall('(\d+\.*\d*)', node_list[7]))
        node["dispersions"] = map(lambda x: float(x),re.findall('(\d+\.*\d*)', node_list[8]))
        node["samples"] = re.findall('"(.*?)"', node_list[11])
        nodes[node["id"]] = node

        if node["id"] == "RT":
            root = node

        # print node

    # print "Done with node construction"
    return nodes, root

def full_tree_construction(node,node_dictionary,counts):
    local_list = []
    local_list.append(node)
    # check_node(node,counts)
    for child in node["children"]:
        local_list.append(full_tree_construction(node_dictionary[child],node_dictionary,counts))
    return local_list

def feature_cov(node):
    covs = []
    for i,dispersion in enumerate(node["dispersions"]):
        try:
            covs.append(dispersion / node["medians"][i])
        except ZeroDivisionError:
            covs.append(0.0)
    return covs

def feature_feature_gain(parent,child):

    parent_covs = feature_cov(parent)
    child_covs = feature_cov(child)

    gains = []

    for i,pcov in enumerate(parent_covs):
        gains.append(pcov - child_covs[i])

    return gains

def crawl_gains(tree,gain_dictionary,header):
    if len(tree) > 1:
        split_feature = tree[0]["feature"]
        for branch in tree[1:]:
            gain = feature_feature_gain(tree[0],branch[0])
            feature_map = map(lambda x: header[int(x)], branch[0]["output_features"])
            for i,feature in enumerate(feature_map):
                if (split_feature,feature) in gain_dictionary:
                    gain_dictionary[(split_feature,feature)].append(gain[i])
                else:
                    gain_dictionary[(split_feature,feature)] = [gain[i],]
            crawl_gains(branch,gain_dictionary,header)
    return gain_dictionary

def crawl_to_leaves(tree):
    local_list = []
    if len(tree) > 1:
        for branch in tree[1:]:
            local_list.extend(crawl_to_leaves(branch))
    else:
        local_list = [tree[0],]
    return local_list

def absolute_gain_frequency(tree):
    root = tree[0]
    leaves = crawl_to_leaves(tree)
    total_gains = []
    for leaf in leaves:
        total_gains.extend(feature_feature_gain(root,leaf))

def tree_level_construction(node,node_dictionary,level,occurence_level_dict):
    local_list = []
    local_list.append(node["feature"])
    # if node["feature"] != "None":
    #     local_list.append(node["split"])
    local_list.append(level)
    if node["feature"] != "None":
        if node["feature"] in occurence_level_dict:
            occurence_level_dict[node["feature"]].append(level)
        else:
            occurence_level_dict[node["feature"]] = [level,]
    for child in node["children"]:
        local_list.append(tree_level_construction(node_dictionary[child],node_dictionary,level+1,occurence_level_dict))
    return local_list


# def tree_translation(tree,header):
#     local_list = []
#     if tree[0] != "None":
#         feature_index = int(tree[0])
#         local_list.append(header[feature_index])
#         for branch in tree[2:]:
#             local_list.append(tree_translation(branch,header))
#
#     return local_list

def median_absolute_deviation(array, drop=True):
    mad = np.ones(array.shape[1])
    for i,feature in enumerate(array.T):

        if drop:
            feature = feature[feature != 0]
        median = np.median(feature,axis=0)
        absolute_deviation = np.abs(feature - (np.ones(feature.shape) * median))
        mad[i] = np.median(absolute_deviation)
    return mad

def drop_median(array):
    dm = np.ones(array.shape[1])
    for i,feature in enumerate(array.T):
        dm[i]=np.median(feature[feature != 0])
    return dm

def name_picker_index(names,shape):

    indecies = map(lambda x: int(x),names)

    # picker = np.zeros(shape,dtype=bool)
    # picker[indecies] = 1
    return indecies

def check_node(node,counts):
    feature_indecies = name_picker_index(node["output_features"],counts.shape[1])
    sample_indecies = name_picker_index(node["samples"],counts.shape[0])

    subsample = counts[sample_indecies]
    subsample = subsample.T[feature_indecies].T

    medians = drop_median(subsample)
    mad = median_absolute_deviation(subsample)

    # print node["feature"]
    # print len(node["samples"])
    # print node["output_features"][:10]
    #
    # print "Medians"
    # print node["medians"][:10]
    # print medians[:10]
    # print medians.shape
    # print "Dispersions"
    # print node["dispersions"][:10]
    # print mad[:10]
    # print mad.shape

print sys.argv

header = np.load(sys.argv[1])

counts = np.loadtxt(sys.argv[2])

gain_map = {}

occurence_level_dict = {}

absolute_gains = []

for tree in sys.argv[3:]:

    tree_dict, root = read_tree(tree,header)

    # tree_level_construction(root,tree_dict,1,occurence_level_dict)

    # node_tree = tree_construction(root,tree_dict)

    full_tree = full_tree_construction(root,tree_dict,counts)

    crawl_gains(full_tree,gain_map,header)

    absolute_gains.extend(absolute_gains(full_tree))

feature_frequency = map(lambda x: (x,len(occurence_level_dict[x])), occurence_level_dict)

feature_score = map(lambda x: (x, reduce(lambda y,z: y + (5./float(z)), occurence_level_dict[x])), occurence_level_dict)

feature_score.sort(key=lambda x: x[1])

print feature_score

for feature in feature_score[-20:]:
    print feature[0]
    print occurence_level_dict[feature[0]]

print "GAIN MAP DEBUG"

print list(gain_map)[:10]
print gain_map.values()[:10]

gain_freq = np.array(reduce(lambda x,y: x + y , gain_map.values(), []))

plt.figure()
plt.hist(gain_freq,bins=30,log=True)
plt.savefig("gains.png")

match_list = []
for value in gain_map:
    for observation in gain_map[value]:
        if observation > .5:
            match_list.append(value)

np.savetxt("match_list.txt",np.array(match_list),fmt='%s')
