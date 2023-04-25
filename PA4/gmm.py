"""# Part 2: Mixture of Gaussians

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
import torch.nn.functional as F
import torch.distributions as D
from scipy.stats import truncnorm

"""### Helper Functions"""

'''
Helper function given by teaching team to load data from data2D.npy.
'''
def load_data():
    X =np.load('data2D.npy')
    valid_batch = int(len(X) / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(len(X))
    np.random.shuffle(rnd_idx)
    val_data = X[rnd_idx[:valid_batch]]
    data = X[rnd_idx[valid_batch:]]

    return data, val_data

'''
This function generates a truncated normal distribution between -ve and 
+ve threshold value
'''
def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

'''
The inputs to the function are:
1] X: N examples within the dataset, stored as a tensor of d=2 coordinates
2] MU: Tensor of k centroids, each element having d=2 coordinates

This function calculates the distance between each example in X and each
centroid in MU, after unsqueeze happens, and returns these pair-wise
distances as a tensor of N x k dimension
'''
def distanceFunc(X, MU):
    # Add an extra dimension at the end to make the shape N x d x 1
    X1 = torch.unsqueeze(X, -1)

    # Add extra dim in the start of transpose to make the shape 1 x d x k
    MU1 = torch.unsqueeze(MU.T, 0)

    # Calculate the squared euclidean-distance between all combinations
    # of input example and cluster center
    pair_dist = torch.sum((X1 - MU1)**2, 1)
    return pair_dist

'''
The inputs to the function are:
1] X: N examples within the dataset, stored as a tensor of 
      d=2 coordinates
2] mu: Tensor of k centroids, each element having d=2 coordinates
3] sigma: Tensor of k values, each representing standard deviation 
          of the kth cluster

This function calculates the log of guassian distributions's pdf 
for each possible pair of centroid and example in the dataset, 
and returns it as a tensor of dimensions N x k
'''
def log_GaussPDF(X, mu, sigma):
    # Get the number of dimensions, here dim will be 2
    dim = X.shape[-1]

    # Create tensor containing pi value
    Pi = torch.tensor(float(np.pi))

    # Squares each value of standard deviation and then transposes
    # to make the shape d x k
    sigma_2 = (torch.square(sigma)).T

    # Get pairwise distance of each centroid-example pair
    diff = distanceFunc(X, mu)

    # The following lines emulate taking the logarithm
    # of gaussian distribution's pdf
    log_PDF = diff / sigma_2              # (x-m)^2 / sigma^2
    log_PDF += dim * torch.log(2 * Pi)    # + log(2*pi)
    log_PDF += dim * torch.log(sigma_2)   # + log(sigma^2)
    log_PDF *= -0.5                       # Finally, * for sqrt and 1/2
    return log_PDF

'''
The inputs to the function are:
1] log_PDF: N x k tensor with log of gaussian pdfs of all 
            example-centroid pairs
2] log_pi: Tensor of k values containing probabilities assigned 
           to each cluster

This function calculates and returns two things:
1] The log of joint probability of each example being assigned 
to each cluster as a tensor of dimensions N x k
2] The log of marginal probability for each of the N examples 
i.e. sum of joint probabilities along each column 
for that particular example. It has N elements.
'''
def log_posterior(log_PDF, log_pi):
    # Calculate log of joint pdf of each point w.r.t. each component
    # of shape N x k
    log_joint = log_PDF + log_pi.T

    # Calculate log of marginal pdf for each point
    # of length N
    log_marginal = torch.logsumexp(log_joint,dim=1)

    return log_joint, log_marginal

"""### Training Helpers"""

def train_gmm(train_data, test_data, k = 5, 
              epoch=1000, init_kmeans=False):
    # Load the data
    X_train = torch.from_numpy(train_data)
    X_test = torch.from_numpy(test_data)

    # Initialize logits depending on value of init_kmeans flag
    # If true, use kmeans to get better than random cluster centroids
    if init_kmeans:
        logits = torch.ones(k, requires_grad=True)
        kmeans = KMeans(n_clusters=k, max_iter=5000, n_init=10)
        kmeans = kmeans.fit(train_data)
        mu = torch.tensor(kmeans.cluster_centers_, requires_grad=True)
        lr = 0.005
    else:
        logits = torch.rand(k, requires_grad=True)
        mu = torch.randn((k,X_train.shape[1]), requires_grad=True)
        lr = 0.005

    # Initialize standard deviations
    sigma = np.abs(truncated_normal((k,1), threshold=1))
    sigma = torch.tensor(sigma,requires_grad=True)
    optimizer = torch.optim.Adam([logits, mu, sigma],
                                 lr=lr,
                                 betas=(0.9, 0.99),
                                 eps=1e-5)
    
    # Train the model
    for i in range(epoch):
        logpi = F.log_softmax(logits, dim=0)
        
        # Get log of gaussian pdfs and then get the marginal pdfs
        log_PDF = log_GaussPDF(X_train, mu, sigma)
        _, log_marginal = log_posterior(log_PDF, logpi)

        # Compute the marginal mean as loss value.
        loss = -log_marginal.mean()

        # Update parameters using back-propagation on the loss fn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    
    # Check performance on Test data
    logpi = F.log_softmax(logits, dim=0)

    # log_GaussPDF and log_posterior
    log_PDF = log_GaussPDF(X_test, mu, sigma)
    log_joint_test, log_marginal = log_posterior(log_PDF, logpi)
    test_loss = -log_marginal.mean()

    # Detach all gradient functions and return final trained values
    test_loss = test_loss.detach().numpy()
    log_joint_test = log_joint_test.detach().numpy()
    pi = torch.exp(logpi).detach().numpy()
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()

    return test_loss, log_joint_test, pi, mu, sigma

"""### Testing Functions"""

def test_GMM(k = 5, init_kmeans=False):
    train_data, test_data = load_data()
    test_loss, log_joint_test, pi, mu, sigma = train_gmm(train_data, 
                                                         test_data, 
                                                         k, 
                                                         init_kmeans=
                                                         init_kmeans)

    index = log_joint_test.argmax(axis=1)
    index = index.reshape(len(index), 1)
    new_X = np.concatenate((test_data, index), axis = 1)

    color_list = ['g', 'b', 'm', 'y', 'c']
    for i in range(len(mu)):
        tmp = new_X[new_X[...,-1] == i]
        plt.scatter(tmp[:,0], tmp[:,1], c=color_list[i])
    plt.scatter(mu[:,0], mu[:,1], s=300, c='r', marker = '+')
