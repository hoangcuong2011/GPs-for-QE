# GPs-for-QE

Code of Hybrid NN with GPs

Requirement
To run the code, you need:

    Tensorflow

    GPflow

    numpy

    sklearn

Data format:

feature1,feature2,...,feature_n,1
feature1,feature2,...,feature_n,0
1: good
0: negative


Running the script:

A deep NN:

    python Classification_baseline.py

Hybrid Network:

    python DKL_with_pre_training_HoangCuong.py

Note: you can choose a right kernel function from line 184
you can select the option of pre-training by editing line 180





