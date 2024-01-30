import numpy as np
import random as rd

# getting a runtime warning when computing new center locations 
import warnings
warnings.filterwarnings('ignore')

def initialize_clusters(X,num_of_clusters,mu):
    # calculate min and max values from X
    min = np.min(X,axis=0)
    max = np.max(X,axis=0)
    cluster = []
    # given a certain number of clusters,
    # randomly initialize them using random.uniform()
    for x in range(num_of_clusters):
        cluster.append(rd.uniform(min,max))
    return cluster

def K_Means(X,num_of_clusters,mu):
    # if mu is empty, randomly initialize them
    if len(mu) == 0:
        mu = np.array(initialize_clusters(X,num_of_clusters,mu))
        # reshape numpy array to specific shape
        mu = mu.reshape(2,1)
    else: mu 
    # add samples to a list of lists, where each list
    # is a center and each item in each list is a sample
    recompute = 0
    old_centers = None
    while np.not_equal(mu,old_centers).any() and recompute < 20:
        assign_points = [[] for x in range(num_of_clusters)]
        for point in X:
            # compute euclidean distance and find the closest center
            # relative to the sample being looked at
            if X.ndim == 1:
                distance = np.sqrt(np.sum((point-mu)**2,axis=0))
            else:
                distance = np.sqrt(np.sum((point-mu)**2,axis=1))
            center_loc = np.argmin(distance)
            assign_points[center_loc].append(point)
        old_centers = mu
        # compute the mean of points at each center to find 
        # values of updated centers
        test = [np.nanmean(points,axis=0) for points in assign_points]
        mu = np.array(test)
        recompute += 1
    print(assign_points)
    return mu
    
