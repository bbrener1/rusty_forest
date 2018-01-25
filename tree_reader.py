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



def read_tree(location):

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
        node["feature"] = node_list[3].split(":")[1]
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
    check_node(node,counts)
    for child in node["children"]:
        local_list.append(full_tree_construction(node_dictionary[child],node_dictionary,counts))
    return local_list

def feature_cov(node):
    covs = []
    for i,dispersion in enumerate(node["dispersions"]):
        covs.append(dispersion / node["medians"][i])
    return covs

def feature_feature_gain(parent,child):

    parent_covs = feature_cov(parent)
    child_covs = feature_cov(child)

    gains = []

    for i,pcov in enumerate(parent_covs):
        gains.append(pcov - child_covs[i])

    return gains

def tree_construction(node,node_dictionary):
    local_list = []
    local_list.append(node["feature"])
    if node["feature"] != "None":
        local_list.append(node["split"])
    for child in node["children"]:
        local_list.append(tree_construction(node_dictionary[child],node_dictionary=node_dictionary))
    return local_list

def tree_translation(tree,header):
    local_list = []
    if tree[0] != "None":
        feature_index = int(tree[0].lstrip("Some(\"").rstrip("\")"))
        local_list.append(header[feature_index])
        for branch in tree[2:]:
            local_list.append(tree_translation(branch,header))

    return local_list

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

    print node["feature"]
    print len(node["samples"])
    print node["output_features"][:10]

    print "Medians"
    print node["medians"][:10]
    print medians[:10]
    print medians.shape
    print "Dispersions"
    print node["dispersions"][:10]
    print mad[:10]
    print mad.shape

header = np.load(sys.argv[2])

counts = np.loadtxt(sys.argv[3])

tree_dict, root = read_tree(sys.argv[1])

node_tree = tree_construction(root,tree_dict)

full_tree = full_tree_construction(root,tree_dict,counts)

print tree_translation(node_tree,header)


# for child in node_tree[1:]:
#     print tree_translation(child,header)
