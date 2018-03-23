import os
import sys
import subprocess as sp
import numpy as np

def main():
    print "Testing"
    prefix = os.getcwd()
    fit_predict_file(prefix+"/testing/iris.drop","zeros","branching","1","10","4","4","4","150","1",output_location=prefix+"/testing/precomputed_trees/iris",features=prefix+"/testing/iris.features")
    fit_predict_file(prefix+"/testing/simple.txt","zeros","branching","1","1","1","1","1","8","1",output_location=prefix+"/testing/precomputed_trees/simple")

    print "And now with feeling"

    for i in range(10):
        fit_predict_file(prefix+"/testing/held_out_counts.txt","zeros","branching","100","10","400","1000","1000","800","50",output_location=prefix+"/impute_test/run" + str(i) ,features=prefix+"/testing/header.txt")

def fit_predict(counts):
    pass

def fit_predict_file(counts,drop_mode,prediction_mode,trees,leaves,in_features,out_features,feature_subsample,sample_subsample,processors,output_location=None,features=None,samples=None,reporting=None):

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

    print str(" ".join(command))

    sp.call(command,stdout=reporting)


def transform_file(counts,prediction_mode,drop_mode,trees,processors,output_location=None,features=None,samples=None,reporting=None):

    prefix = "./target/debug/"
    command = [prefix+"forest_prot",]

    command.append("predict")
    command.extend(["-c",counts])
    command.extend(["-m",prediction_mode])
    command.extend(["-d",drop_mode])
    command.extend(["-tg",trees])
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

    print str(" ".join(command))

    sp.call(command,stdout=reporting)

def fit_file(counts,drop_mode,trees,leaves,in_features,out_features,feature_subsample,sample_subsample,processors,output_location=None,features=None,samples=None,reporting=None):

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
    command.extend(["-o", output_location])

    if features != None:
        command.extend(["-f",features])

    if samples != None:
        command.extend(["-s",samples])

    print str(" ".join(command))

    sp.call(command,stdout=reporting)

if __name__ == "__main__":
    main()
