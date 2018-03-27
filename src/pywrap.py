import os
import glob
import sys
import subprocess as sp
import numpy as np
import re

def main():
    print "Testing"
    prefix = os.getcwd()
    fit_predict_file(counts=prefix+"/testing/iris.drop",drop_mode="zeros",prediction_mode="branching",trees="1",leaves="10",in_features="4",out_features="4",feature_subsample="4",sample_subsample="150",processors="1",output_location=prefix+"/testing/precomputed_trees/iris",features=prefix+"/testing/iris.features",reporting=open(os.devnull,mode='w'))

    fit_predict_file(prefix+"/testing/simple.txt",sample_subsample="8",output_location=prefix+"/testing/precomputed_trees/simple",reporting=open(os.devnull,mode='w'))

    print "Loading multiprocessor test:"

    iris = np.loadtxt(prefix+"/testing/iris.drop")

    print iris[:10]

    print fit_predict(iris,processors="20",in_features="4",out_features="4",feature_subsample="4",sample_subsample="150",reporting=open(os.devnull,mode='w'))

    indicator_file = open('./working/indicator.txt',mode='w')

    indicator_file.write("Done with multiprocessor testing\n")

    # print "And now with feeling"
    #
    # for i in range(1):
    #     report_file = open(prefix+"/wrapper_test/run." + str(i) + ".log",mode='w')
    #     trees = fit_file(prefix+"/testing/held_out_counts.txt","zeros","branching","1000","100","400","1000","1000","800","50",output_location=prefix+"/wrapper_test/run." + str(i) ,reporting=report_file)
    #     predict_file(prefix+"/testing/held_out_counts.txt",trees)
#
# features=prefix+"/testing/header.txt"


def fit_predict(counts,drop_mode="zeros",prediction_mode="branching",trees="1",leaves="1",in_features="1",out_features="1",feature_subsample="1",sample_subsample="1",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):

    tree_files = fit(counts,drop_mode=drop_mode,prediction_mode=prediction_mode,trees=trees,leaves=leaves,in_features=in_features,out_features=out_features,feature_subsample=feature_subsample,sample_subsample=sample_subsample,processors="10",output_location=output_location,features=features,samples=samples)

    return predict(counts,tree_files)

def fit(counts,drop_mode="zeros",prediction_mode="branching",trees="1",leaves="1",in_features="1",out_features="1",feature_subsample="1",sample_subsample="1",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):

    if output_location == "./working/temp":
        for file in (glob.glob("./working/*")):
            os.remove(file)

    np.savetxt("./working/counts.txt", counts)

    fit_file("./working/counts.txt",drop_mode=drop_mode,trees=trees,leaves=leaves,in_features=in_features,out_features=out_features,feature_subsample=feature_subsample,sample_subsample=sample_subsample,processors=processors,output_location=output_location,features=features,samples=samples)

    tree_files = glob.glob('./run\.[0-9]+\.[0-9]+')

    print tree_files

    return tree_files,counts

def predict(counts,trees,prediction_mode="branching",drop_mode="zeros",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):
    np.savetxt("./working/counts.txt",counts)

    predict_file("./working/counts.txt",trees,prediction_mode=prediction_mode,drop_mode=drop_mode,processors=processors,output_location=output_location,features=features,reporting=reporting)

    return np.loadtxt("./working/temp.predictions")


def fit_predict_file(counts,drop_mode="zeros",prediction_mode="branching",trees="1",leaves="1",in_features="1",out_features="1",feature_subsample="1",sample_subsample="1",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):

    prefix = "./target/release/"
    command = [prefix+"forest_prot",]

    command.append("construct_predict")
    command.extend(["-c",counts])
    command.extend(["-d",drop_mode])
    command.extend(["-m",prediction_mode])
    command.extend(["-t",trees])
    command.extend(["-l",leaves])
    command.extend(["-p",processors])
    command.extend(["-if",in_features])
    command.extend(["-of",out_features])
    command.extend(["-fs",feature_subsample])
    command.extend(["-ss",sample_subsample])
    command.extend(["-o", output_location])

    if features != None:
        command.extend(["-f",features])

    if samples != None:
        command.extend(["-s",samples])

    sp.Popen(command,stdout=reporting)


def predict_file(counts,trees,prediction_mode="branching",drop_mode="zeros",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):

    prefix = "./target/debug/"
    command = [prefix+"forest_prot",]

    command.append("predict")
    command.extend(["-c",counts])
    command.extend(["-m",prediction_mode])
    command.extend(["-d",drop_mode])
    command.extend((["-tg"] + trees))
    command.extend(["-p",processors])
    command.extend(["-if",in_features])
    command.extend(["-of",out_features])
    command.extend(["-fs",feature_subsample])
    command.extend(["-ss",sample_subsample])
    command.extend(["-o", output_location])

    if features != None:
        command.extend(["-f",features])

    if samples != None:
        command.extend(["-s",samples])

    print "Transforming counts with some trees"

    print str(" ".join(command))

    sp.Popen(command,stdout=reporting)

def fit_file(counts,drop_mode="zeros",trees="1",leaves="1",in_features="1",out_features="1",feature_subsample="1",sample_subsample="1",processors="1",output_location="./working/temp",features=None,samples=None,reporting=None):


    prefix = "./target/debug/"
    command = [prefix+"forest_prot",]

    command.append("construct_predict")
    command.extend(["-c",counts])
    command.extend(["-d",drop_mode])
    command.extend(["-t",trees])
    command.extend(["-l",leaves])
    command.extend(["-p",processors])
    command.extend(["-if",in_features])
    command.extend(["-of",out_features])
    command.extend(["-fs",feature_subsample])
    command.extend(["-ss",sample_subsample])


    if features != None:
        command.extend(["-f",features])

    if samples != None:
        command.extend(["-s",samples])

    print "Fitting to a file"

    print str(" ".join(command))

    children = []

    if processors > 10:

        for i in range(int(processors)/10):

            stock_copy = command[:]
            stock_copy.extend(["-p", "10"])
            stock_copy.extend(["-o", output_location + "." + str(i)])

            children.append(sp.Popen(stock_copy,stdout=reporting))

    else:

        command.extend(["-p",processors])
        command.extend(["-o", output_location])

        children.append(sp.Popen(command,stdout=reporting))

    for child in children:
        child.wait()

    # tree_files = glob.glob(output_location+".[0-9]+.[0-9]+")

    return tree_files


if __name__ == "__main__":
    main()
