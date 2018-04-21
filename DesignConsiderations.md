# Design Condisderations for the Median Forest Program

This software is intended to perform several functions:

It is capable of performing imputation on missing data in order to calculate appropriate values for such. 

It is capable of performing multivariate regression on samples based on limited feature inputs.

It is capable of reporting back on feature relationships and interactions in the regression. 

The underlying function of the software is based on an implementation of decision tree based regression, both gradient boosted and plain random forest.

The random forest implementation is able to ignore dropped values during regression, and performs splits of data by minimizing Median Absolute Deviation from the Median (hereafter MAD) L1, L2, or .5L, the fractional L norm. 

Gradient boosting can be performed by summing error matrices or by over-representing error-prone features and samples. 

To aid interpretation, the program provides certain summary outputs that allow you to determine if appropriate norming, boosting, and ensemble integration was performed. 

The purpose of this program is interpretation of single-cell RNA Seq data, but it is appropriate for interpretation of other high-dimensional data. 

The general structure of the program is as follows:

The outer-most class is the Random Forest



# Random Forests:

Random Forest contains:
     - The matrix to be analyzed
     - Decision Trees
     - Decision Tree Thread Pool

     - Important methods:
         - Method that generates decision trees and calls on them to grow branches
         - Method that generates predicted values for a matrix of samples





 Trees:

 Trees contain:
     - Root Node
     - Feature Thread Pool Sender Channel
     - Drop Mode

 Each tree contains a subsampling of both rows and columns of the original matrix. The subsampled rows and columns are contained in a root node, which is the only node the tree has direct access to.


 Feature Thread Pool:

 Feature Thread Pool contains:
     - Worker Threads
     - Reciever Channel for jobs

     - Important methods:
         - A wrapper method to compute a set of medians and MADs for each job passed to the pool. Core method logic is in Rank Vector

 Feature Thread Pools are containers of Worker threads. Each pool contains a multiple in, single out channel locked with a Mutex. Each Worker contained in the pool continuously requests jobs from the channel. If the Mutex is unlocked and has a job, a Worker thread receives it.

     Jobs:
         Jobs in the pool channel consist of a channel to pass back the solution to the underlying problem and a freshly spawned Rank Vector (see below). The job consists of calling a method on the RV that consumes it and produces the medians and Median Absolute Deviations (MAD) from the Median of the vector if a set of samples is removed from it in a given order. This allows us to determine what the Median Absolute Deviation from the Median would be given the split of that feature by some draw order. The draw orders given to each job are usually denoting that the underlying matrix was sorted by another feature.

 Worker threads are simple anonymous threads kept in a vector in the pool, requesting jobs on loop from the channel.
