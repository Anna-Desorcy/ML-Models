import numpy as np

def split_data(X,Y):
    # 80% train and 20% test
    X_train, X_test = np.split(X,[int(0.8 * len(X))])
    Y_train, Y_test = np.split(Y,[int(0.8 * len(Y))])
    return X_train,X_test,Y_train,Y_test

def KNN_predict(X_train,X_test,Y_train,K):
    k_points = []
    # calculate euclidean distance
    distance = np.sqrt(np.sum((X_test-X_train)**2,axis=1))
    # sort by distance from label value to point
    for dist,label in sorted(zip(distance,Y_train)):
        k_points.append(label)
    k_points = k_points[:K]
    # calculate the most common element in k_points
    element = max(set(k_points),key=k_points.count)
    return element

def KNN_test(X,Y,q,z,K):
    # splitting into train/test
    X_train,X_test,Y_train,Y_test = split_data(X,Y)
    prediction = []
    for test in X_test:
        prediction.append(KNN_predict(X_train,test,Y_train,K))
    correct = []
    for i in range(len(Y_test)):
        if prediction[i] == Y_test[i]:
            correct.append(prediction)
    acc = len(correct)/len(Y_test)
    return acc
    
# Step 1 - Assign a value to K.

# Step 2 - Calculate the distance between the new data entry 
#          and all other existing data entries. Arrange them 
#          in ascending order. [LINES 13-16]

# Step 3 - Find the K nearest neighbors to the new entry based 
#          on the calculated distances. [LINE 17]

# Step 4 - Assign the new data entry to the majority class in 
#          the nearest neighbors. [LINE 19]