# Internal organization:

Main creates a Random Forest

Random Forest contains a prototype Tree and Trees derived from it

The prototype tree spawns a Thread Pool on creation, and passes channels to it to any child trees

A threadpool spawns some number of workers on creation and owns them indefinitely, also owns a mpsc channel for receiving jobs.

Workers grab jobs off a mutex-locked mpsc channel for the thread pool and send information back through enclosed channels.


Each Tree contains a rank-table containing rank-vectors for each feature contained in the tree.

Each rank vector contains methods for finding medians and median distances for the feature at any given time and methods for permuting the data structure

Each rank vector contains a raw vector, raw vectors behave similarly to linked lists containing a node for each sample contained in the tree.

The ordering of the samples in the raw vector is uniform between all raw vectors in a node, so telling a ranked vector to pop sample index x will pop the correct sample out of each raw vector


