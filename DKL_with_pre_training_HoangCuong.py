
import os
import copy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
import numpy as np
from sklearn import cluster
from scipy.spatial import distance
import pandas as pd
from keras.utils import np_utils
import gpflow as gpf
from sklearn.metrics import f1_score


class DataPlaceholders(object):
    def __init__(self):
        self.data = tf.placeholder(tf.float32)        
        self.keep_prob = tf.placeholder(tf.float32)        
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="labels")



def create_bias(shape, initial_val=0.1, dtype=tf.float32):
    initial = tf.constant(initial_val, shape=shape, dtype=dtype, name="bias")
    return initial



def make_feedforward_nn(x_placeholder, end_h):
    W1 = tf.get_variable("W1", shape=[614, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", initializer=create_bias([512]))
    h1 = tf.nn.relu(tf.matmul(x_placeholder, W1) + b1)
    W2 = tf.get_variable("W2", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", initializer=create_bias([256]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    W3 = tf.get_variable("W3", shape=[256, end_h], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", initializer=create_bias([end_h]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)


    return h3


def suggest_good_intial_inducing_points(phs: DataPlaceholders, x_data, h, tf_session, num_inducing):
    h_data = tf_session.run(h, feed_dict={phs.data: x_data, phs.keep_prob: 1.0})
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_inducing, batch_size=num_inducing*10)
    kmeans.fit(h_data)
    new_inducing = kmeans.cluster_centers_
    return new_inducing


def suggest_sensible_lengthscale(phs: DataPlaceholders, x_data, h, tf_session):
    h_data = tf_session.run(h, feed_dict={phs.data: x_data, phs.keep_prob: 1.0})
    lengthscale = np.mean(distance.pdist(h_data, 'euclidean'))
    return lengthscale



def standardize_data(X_train, X_test, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid



def standardize_data(X_train, X_test2016, X_test2017, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test2016 -= X_mean
    X_test2016 /= X_std

    X_test2017 -= X_mean
    X_test2017 /= X_std


    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test2016, X_test2017, X_valid

def compute_scores(flat_true, flat_pred):
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)

def resampleFile():
    filename = open("train.revised", "w")
    file = open("train", "r")
    for x in file:
        x = x.strip()
        filename.write(x+"\n")
        if x.endswith(",0"):
            #filename.write(x+"\n")
            filename.write(x+"\n")
            filename.write(x+"\n")
            
    filename.close()
    file.close()



def main():
    """
    Simple demonstration of how you can put a GP on top of a NN and train the whole system end-to-end in GPflow-1.0.


    Note
    that in the new GPflow there are new features that we do not take advantage of here but could be used to make
    the whole example cleaner. For example you may want to use a gpflow.train Optimiser as this will take care of
    passing in the GP model feed dict for you as well as initially initialising the optimisers TF variables.
    You could also choose to tell the gpmodel to initialise the NN variables by subclassing SVGP and overriding the
    appropriate variable initialisation method.
    """
    # ## We load in the MNIST data. We will create a validation set but will not use it in this simple example.
    
    dataset = np.loadtxt("test", delimiter=",")
    x_test_2016 = dataset[:,0:614]
    y_test_2016 = dataset[:,614].reshape(-1,1)

    dataset = np.loadtxt("test", delimiter=",")
    x_test_2017 = dataset[:,0:614]
    y_test_2017 = dataset[:,614].reshape(-1,1)
    #print(x_test[20])

    dataset = np.loadtxt("dev", delimiter=",")
    x_valid = dataset[:,0:614]
    y_valid = dataset[:,614].reshape(-1,1)
    #resampleFile()
    dataset = np.loadtxt("train.revised", delimiter=",")
    x_train = dataset[:,0:614]
    y_train = dataset[:,614].reshape(-1,1)

    
    x_train_root = x_train
    x_valid_root = x_valid
    x_train, x_test_2016, x_test_2017, x_valid = standardize_data(copy.deepcopy(x_train_root), x_test_2016, x_test_2017, copy.deepcopy(x_valid_root))


    #rng = np.random.RandomState(100)
    #train_permute = rng.permutation(x_train.shape[0])
    #x_train, y_train = x_train[train_permute, :], y_train[train_permute, :]

    # ## We set up a TensorFlow Graph and a Session linked to this.
    tf_graph = tf.Graph()
    tf_session = tf.Session(graph=tf_graph)

    # ## We have some settings for the model and its training which we will set up below.
    num_h = 17
    num_classes = 2 #could be improved here
    num_inducing = 100
    minibatch_size = 250

    # ## We set up the NN part of the GP kernel. This needs to be put on the same graph
    with tf_graph.as_default():

        phs = DataPlaceholders()
        h = make_feedforward_nn(phs.data, num_h)
        h = tf.cast(h, gpf.settings.tf_float)

        nn_vars = tf.global_variables()  # only nn variables exist up to now.
        saver = tf.train.Saver(max_to_keep = 1000)
        

    tf_session.run(tf.variables_initializer(nn_vars))        
    #saver.restore(tf_session, "baselinemodel_at_epoch"+str(2)+".ckpt")

    # ## We now set up the GP part. Instead of the usual X data it will get the data after being processed by the NN.
    with gpf.defer_build():
        kernel = gpf.kernels.Matern52(num_h)
        likelihood = gpf.likelihoods.MultiClass(num_classes)
        gp_model = gpf.models.SVGP(h, phs.label, kernel, likelihood, np.ones((num_inducing, num_h), gpf.settings.np_float),
                               num_latent=num_classes, whiten=False, minibatch_size=None, num_data=x_train.shape[0])
    # ^ so we say minibatch size is None to make sure we get DataHolder rather than minibatch data holder, which
    # does not allow us to give in tensors. But we will handle all our minibatching outside.
    gp_model.compile(tf_session)

    # ## The initial lengthscales and inducing point locations are likely very bad. So we use heuristics for good
    # initial starting points and reset them at these values.

    SEED = 448
    np.random.seed(SEED)
    train_unique = np.unique(x_train, axis=0)
    shuffle = np.arange(len(train_unique))
    np.random.shuffle(shuffle)    
    x_train_unique_shuffle = train_unique[shuffle]

    gp_model.feature.Z.assign(suggest_good_intial_inducing_points(phs, x_train_unique_shuffle[:5000, :], h, tf_session, num_inducing), tf_session)
    gp_model.kern.lengthscales.assign(suggest_sensible_lengthscale(phs, x_train_unique_shuffle[:5000, :], h, tf_session) 
        + np.zeros_like(gp_model.kern.lengthscales.read_value()), tf_session)

   

    # ^ note that this assign should reapply the transform for us :). The zeros ND array exists to make sure
    # the lengthscales are the correct shape via  broadcasting

    # ## We create ops to measure the predictive log likelihood and the accuracy.
    with tf_graph.as_default():
        log_likelihood_predict = gp_model.likelihood.predict_density(*gp_model._build_predict(h), phs.label)
        outputs = tf.argmax(gp_model.likelihood.predict_mean_and_var(*gp_model._build_predict(h))[0], axis=1, output_type=tf.int32)
        accuracy = tf.cast(tf.equal(outputs, tf.squeeze(phs.label)), tf.float32)
        avg_acc = tf.reduce_mean(accuracy)
        avg_ll = tf.reduce_mean(log_likelihood_predict)

        # ## we now create an optimiser and initialise its variables. Note that you could use a GPflow optimiser here
        # and this would now be done for you.
        all_vars_up_to_trainer = tf.global_variables()
        optimiser = tf.train.AdamOptimizer(1e-4)
        print(tf.global_variables())
        minimise = optimiser.minimize(gp_model.objective)  # this should pick up all Trainable variables.
        adam_vars = list(set(tf.global_variables()) - set(all_vars_up_to_trainer))
        neg_gp_opj = -gp_model.objective
        tf_session.run(tf.variables_initializer(adam_vars))
        saver = tf.train.Saver(max_to_keep = 1000)







    # ## We now go through a training loop where we optimise the NN and GP. we will print out the test results at
    # regular intervals.    
    results = []    
    SEED = 449
    np.random.seed(SEED)
    for i in range(20): #100 epochs
        if i>0:
            #load truoc
            #print_tensors_in_checkpoint_file(file_name="model_at_epoch"+str(i-1)+".ckpt", tensor_name='', all_tensors=False)

            saver.restore(tf_session, "mmodel_at_epoch"+str(i-1)+".ckpt")

            fd = gp_model.feeds or {}
            fd.update({phs.keep_prob: 1.0, phs.data: x_valid,
                       phs.label: y_valid})
            accuracy_evald, log_like_evald, outputs_evald = tf_session.run([avg_acc, avg_ll, outputs], feed_dict=fd)
            print("Epoch {}: \Dev set LL {}, Acc {}, Outputs {}".format(i, log_like_evald, 
                accuracy_evald, outputs_evald))
            outputs_evald = outputs_evald.reshape(-1, 1)
            #print(outputs_evald)
            print("Result from the previous epoch on dev:")
            compute_scores(y_valid, outputs_evald)

            
            fd = gp_model.feeds or {}
            fd.update({phs.keep_prob: 1.0, phs.data: x_test_2016,
                       phs.label: y_test_2016})
            accuracy_evald, log_like_evald, outputs_evald = tf_session.run([avg_acc, avg_ll, outputs], feed_dict=fd)
            print("Epoch {}: \nTest set LL {}, Acc {}, Outputs {}".format(i, log_like_evald,
                accuracy_evald, outputs_evald))
            outputs_evald = outputs_evald.reshape(-1, 1)
            #print(outputs_evald)
            print("Result from the previous epoch on test:")
            compute_scores(y_test_2016, outputs_evald)


            fd = gp_model.feeds or {}
            fd.update({phs.keep_prob: 1.0, phs.data: x_test_2017,
                       phs.label: y_test_2017})
            accuracy_evald, log_like_evald, outputs_evald = tf_session.run([avg_acc, avg_ll, outputs], feed_dict=fd)
            print("Epoch {}: \nTest set LL {}, Acc {}, Outputs {}".format(i, log_like_evald,
                accuracy_evald, outputs_evald))
            outputs_evald = outputs_evald.reshape(-1, 1)
            #print(outputs_evald)
            print("Result from the previous epoch on test:")
            compute_scores(y_test_2017, outputs_evald)


        shuffle = np.arange(y_train.size)        
        np.random.shuffle(shuffle)
        print(shuffle)
        x_train_shuffle = x_train[shuffle]
        y_train_shuffle = y_train[shuffle]
        data_indx = 0
        while data_indx<y_train.size:
            lastIndex = data_indx + minibatch_size
            if lastIndex>=y_train.size:
                lastIndex = y_train.size
            indx_array = np.mod(np.arange(data_indx, lastIndex), x_train_shuffle.shape[0])
            #print("array", indx_array)
            data_indx += minibatch_size
            #print(data_indx)fz
            fd = gp_model.feeds or {}
            fd.update({
                phs.keep_prob: 1.0,
                phs.data: x_train_shuffle[indx_array],
                phs.label: y_train_shuffle[indx_array]
                })
            _, loss_evd = tf_session.run([minimise, neg_gp_opj], feed_dict=fd)
            # Print progress every 1 epoch.
        save_path = saver.save(tf_session, "./mmodel_at_epoch"+str(i)+".ckpt")

    print("Done!")



if __name__ == '__main__':
    main()