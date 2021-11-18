# Spot Virtual Machine Eviction Prediction

This repository contains the source code for the implementation of the cluster-level and node-level eviction prediction algorithms.

## Source Structure

* `./dataprocess.py`: defines dataset classes and preprocess dataset in ./data/;
* `./model.py`: contains baseline models and our spatial-temporl node-level model;
* `./train.py`: trains all baselines and our spatial-temporl node-level model;
* `./evaluation.py`: evaluate baselines and our model on the test dataset in ./test/;
