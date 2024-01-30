import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X,Y):
    # grab number of features from data
    num_of_features = X.shape[1]
    # initialize numpy array of correct length with zeros
    weights = np.zeros(num_of_features)
    # initialize bias as zero with correct shape for
    # later on in calculations
    bias = np.array(0).reshape(1,)
    epoch = 0
    # calculate activation of each sample 
    # update the weights an bias if activation is <= 0
    # return the weights and bias 
    for sample,label in zip(X,Y):
        activation = (np.dot(weights,sample) + bias)
        if (activation*label) <= 0:
            weights = weights + label*sample
            bias += label
        epoch += 1
    W = [weights,bias]
    return W

def perceptron_test(X,Y,weights,bias):
    correct = 0
    total = 0
    # based on the weights and bias we calculated before
    # compute the activation with those and find out
    # how many test samples we got correct
    # return the accuracy
    for sample,label in zip(X,Y):
        activation = (np.dot(weights,sample)+bias)
        if activation > 0 and label > 0:
            correct += 1
        elif activation < 0 and label < 0:
            correct += 1
        total += 1
    acc = correct / total
    
    return acc