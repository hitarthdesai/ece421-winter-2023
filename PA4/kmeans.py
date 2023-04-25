# Part 1: K-Means Clustering

### Import Modules
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn

"""### Helper Functions"""

'''
Helper function given by teaching team to load data from data2D.npy.
'''
def load_data():
    X =np.load('data2D.npy')
    valid_batch = int(len(X) / 4.0)
    np.random.seed(45689)
    rnd_idx = np.arange(len(X))
    np.random.shuffle(rnd_idx)
    val_data = X[rnd_idx[:valid_batch]]
    train_data = X[rnd_idx[valid_batch:]]
    
    return train_data, val_data

"""### Training Helpers"""

'''
This function helps us train the kmeans algorithm using given 
value for k, for given number of epochs.
It returns the final training loss value and cluster centroids.
'''
def train_kmean_torch(train_data, k = 5, lr=0.1, epoch=150):
    # List of k random cluster centers containing d=2 coordinates
    m = torch.rand((k, train_data.shape[1]),
                   requires_grad=True,
                   dtype=torch.float64)

    # Convert training data to tensor
    X_train = torch.from_numpy(train_data)

    # Define optimizer as ADAM optimizer with iterable list of centroids
    optimizer = torch.optim.AdamW([m], lr=lr)

    # For each epoch, do the follwing
    for e in range(epoch):
        list_mse = []

        # For kth centroid, calculate the squared differences 
        # between the two coordinates of centroid and each 
        # training example. Then, take the mean of the two 
        # squared differences to get the mse
        for i in range(k):
            differences = nn.functional.mse_loss(X_train,
                                                 m[i].expand_as(X_train),
                                                 reduction='none')
            list_mse.append(torch.sum(differences, dim=1,
                                      dtype=torch.float64))
            
        # Stack the list of lists as a k x N tensor
        list_mse_torch = torch.stack(list_mse, dim=0)

        # Calculate the minimum mse along the column of each example
        # Signifies the centroid to which given example is closest
        # We now get a N sized tensor
        list_mse_torch_min,_ = torch.min(list_mse_torch, dim=0)

        # Take mean of tensor to get loss function of kmeans algo
        L_train = torch.mean(list_mse_torch_min, dtype=torch.float64)
        
        # Perform backward propagation
        optimizer.zero_grad()
        L_train.backward()
        optimizer.step()

    # After all epochs are done, detach the gradient of loss and centroids
    L_train = L_train.detach().numpy()
    m = m.detach().numpy()

    return L_train, m

'''
Given a dataset to test and list of centroids, it calculates average loss
'''
def evaluate(test_data, m):
    # Initialize number of examples and loss value
    num_examples = len(test_data)
    loss = 0.0

    # Perform the following for every example in the test dataset:
    # Calculate mse of the point from each centroid, then from this,
    # Take the minimum and add it to the loss value
    for example in test_data:
        mse_list = [np.sum((example-centroid)**2) for centroid in m]
        loss += np.min(mse_list)

    # Finally return average loss over the entire dataset
    return loss/num_examples

def get_association(test_data, m):
    # Get number of centroids, examples, dimensions and initialize losses
    num_cluster = len(m)
    N, d = test_data.shape
    L_k = np.zeros((N, num_cluster))

    # For each cluster, calculate mse of each point in test dataset
    for k in range(num_cluster):
        L_k[:,k] = [np.sum((example-m[k])**2) for example in test_data]
    
    #Assign to the nearest cluster.
    index = np.argmin(L_k, axis = -1)
    index = index.reshape(len(index), 1)
    return index

"""### Testing Functions"""

'''
This testing function is given by teaching team to test our
implementation of kmeans algorithm
'''
def test_pytorch(train_data, test_data, k=5):
    L,m = train_kmean_torch(train_data, k)
    index = get_association(test_data, m)
    new_X = np.concatenate((test_data, index), axis = 1)

    print ("PyTorch test score:", evaluate(test_data, m))

    color_list = ['g', 'b', 'm', 'y', 'c']
    for i in range(len(m)):
        tmp = new_X[new_X[...,-1] == i]
        plt.scatter(tmp[:,0], tmp[:,1], c=color_list[i])

'''
This testing function is given by teaching team to scikit implementation
of kmeans algorithm
'''
def test_sckitlearn(train_data, test_data, k=5):
    kmeans = KMeans(n_clusters=k, max_iter=5000,
                    algorithm='lloyd', n_init=10)
    kmeans = kmeans.fit(train_data)

    index = kmeans.predict(test_data)
    index = index.reshape(len(index), 1)
    new_X = np.concatenate((test_data, index), axis = 1)

    print ("Scikit-learn test score:",
           evaluate(test_data, kmeans.cluster_centers_))

    color_list = ['g', 'b', 'm', 'y', 'c']
    for i in range(len(kmeans.cluster_centers_)):
        tmp = new_X[new_X[...,-1] == i]
        plt.scatter(tmp[:,0], tmp[:,1], c=color_list[i])