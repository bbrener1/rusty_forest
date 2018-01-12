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
    tree_nodes = tree_string.split("ID:")
    nodes = []
    for node_string in tree_nodes:
        node_list = node_string.split("\n")
        node = {}
        node["id"] = node[0].split(":")
        node["parent"] = node[1].split(":")[1:]
        node["feature"] = node[2].split(": ")[1]
        node["split"] = float(node[3].split(":")[1])
        node["output_features"] = map(lambda x: float(x),re.findall('"(.*?)"', node[5]))
        node["medians"] = map(lambda x: float(x),re.findall('(\d*\.*\d*)', node[6]))
        node["dispersions"] = map(lambda x: float(x),re.findall('(\d*\.*\d*)', node[7]))
        node["samples"] = map(lambda x: float(x),re.findall('"(.*?)"', node[10]))
        nodes.append(node)

        print node

read_tree(sys.argv[1])
