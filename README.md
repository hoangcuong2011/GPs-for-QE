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
where
    
    1: good
    
    0: bad


Running the script:

A deep NN:

    python Classification_baseline.py

Hybrid Network:

    python DKL_with_pre_training_HoangCuong.py

Note: you can choose a right kernel function from line 184

you can select the option of pre-training by editing line 180



The code is mainly relied on the code provided by https://github.com/john-bradshaw: (see this: https://github.com/GPflow/GPflow/issues/505#issuecomment-377187887)

Any comments are welcome. Please contact hoangcuong2011@gmail.com
