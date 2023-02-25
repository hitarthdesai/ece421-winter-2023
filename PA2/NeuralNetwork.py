"""
Import Modules
"""

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 

"""
Part 1: Activation Functions, Output Functions and Error Functions
"""

'''
Function that emulates the ReLU activation function.
Returns 0 for negative argument, otherwise linear.
'''
def activation(s):
    return 0 if s<0 else s

'''
Function that returns derivative of ReLU activation function.
Returns 1 for positive and 0 otherwise.
Although should be undefined at s=0, its value is assumed to
be 0 for simplicity in calculation.
'''
def derivativeActivation(s):
    return 1 if s>0 else 0

'''
Calculates the sigmoid (logistic regression) output of input
'''
def outputf(s):
    return 1/(1+np.exp(-s))

'''
Calculates derivative of the sigmoid function at argument
'''
def derivativeOutput(s):
    sigm=outputf(s)
    return sigm*(1-sigm)

'''
Calculates error based on true class of the example whose prediction
probability is given inside x_L
'''
def errorf(x_L,y):
    if y==1: return -np.log(x_L)
    if y==-1: return -np.log(1-x_L)

'''
Calculates the derivative of the error for given probability and
true label
'''
def derivativeError(x_L,y):
    if y==1: return -1/x_L
    if y==-1: return 1/(1-x_L)

"""
Part 2: Training the Neural Network
"""

'''
Function to calculate prediction for one example
based on provided weight matrices for all layers.
Both x and weights account for bias term.

Output variables from the function:
1] X: Output of each node in the network's layers (i, hidden_layers, o)
2] S: Input to each node of network's layers except i
'''
def forwardPropagation(x, weights):
    # Total number of layers in network
    L=len(weights)+1
    X=[list()]*L
    S=[list()]*(L-1)

    # current_x represents which layer's output is being considered
    current_x=x
    X[0]=x

    for l in range(L-1):
        # Calculate the output and assign it inside S
        weight_matrix=weights[l]
        output=np.matmul(current_x, weight_matrix)
        S[l]=output

        # Do first block if current layer is not output layer
        # Otherwise, do the second block
        if l!=L-2:
            activated_output=[activation(p) for p in output]
            current_x=np.hstack((1,activated_output))
        else:
            current_x=[outputf(p) for p in output]

        # Store the activated output 
        X[l+1]=current_x
        
    return X, S

'''
Function that returns the error value for given
output layer and truth value
'''
def errorPerSample(X,y_n):
    return errorf(X[-1][0], y_n)

'''
Function for the calulation of gradients for each node using the
Back-Propagation algorithm for given node outputs/inputs, labels
and weight matrices.

Outputs list of gradients for each node in network's layers
'''
def backPropagation(X,y_n,s,weights):
    # Number of layers in the network (i, hidden_layers, o)
    L=len(X)
    gradients=[]

    deltas=[]
    
    # Calculate delta for the junction between layer L and L-1
    output_delta=derivativeError(X[-1][0],y_n)*derivativeOutput(s[-1])
    deltas.insert(0,output_delta)

    # Calculate delta for the junction between layer L-1 and L-2, and so on
    for l in range(L-3, -1, -1):
        # Get weight for the next junction and remove the column containing
        # the bias terms because they do not contribute to gradient
        next_w=weights[l+1]
        weights_without_bias=np.delete(next_w, (0), axis=0)

        # Get delta for the next junction and get the product of corresponding
        # node weight and delta
        next_d=deltas[0]
        current_d=np.dot(weights_without_bias, next_d)

        # Get the node input values for current layer and activate them
        current_input=s[l]
        actv_der=[derivativeActivation(p) for p in current_input]

        # Multiply each activated input with corresponding node delta
        current_d=(current_d.T*actv_der).T
        deltas.insert(0,current_d)

    # Calculate gradient for each junction using the junction deltas        
    for l in range(len(deltas)):
        # Get the deltas for the current junction and the output for 2nd layer
        current_d=deltas[l]
        current_x=X[l]

        # Get product of every combination between node deltas and outputs
        gradient=np.outer(current_x, current_d)
        gradients.append(gradient)
    return gradients

'''
Function that uses the weight update equation to update given weights based
on provided gradients and learning_rate(alpha)

Returns list of new weights for each junction
'''
def updateWeights(weights,g,alpha):
    new_weights=[]
    for i in range(len(weights)):
        # For each junction, multiply the gradient with alpha and subtract
        # from original weights to get new weights
        new_weight=weights[i]-alpha*g[i]
        new_weights.append(new_weight)
    return new_weights

'''
Function used to train network of given hidden_layer specs with specified
learning_rate(alpha) for given number of epochs

Returns list of error over all epochs and weights of trained network
'''
def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    # Get details of the network and training data
    num_hidden_layers=len(hidden_layer_sizes)
    num_examples,num_features=X_train.shape

    weights=[]
    for l in range(num_hidden_layers+1):
        weight_l=[]
        if l==0: # Weight matrix for input layer
            weight_l=np.random.normal(0, 0.1, (1+num_features,hidden_layer_sizes[0]))
        elif l<num_hidden_layers: # Weight matrix for hidden layers
            weight_l=np.random.normal(0, 0.1, (hidden_layer_sizes[l-1]+1,hidden_layer_sizes[l]))
        else: # Weight matrix for output layer
            weight_l= np.random.normal(0, 0.1, (hidden_layer_sizes[-1]+1,1)) 

        weights.append(weight_l)

    # Add column of ones for bias term to trianing dataset
    X_train=np.hstack((np.ones((num_examples,1)),X_train))

    # Initialize the list of epoch errors
    err=np.zeros((epochs,1))
    
    for epoch in range(epochs):
        # List of randomly shuffled indices for accessing example-label pairs
        indices=np.arange(0, num_examples)
        np.random.shuffle(indices)

        for n in range(num_examples):
            # Get the random indice, and corresponding example-label pair
            index=indices[n]
            example=X_train[index].T
            label=y_train[index]

            # Finish one pass over given example
            X,S=forwardPropagation(example, weights)
            err[epoch]+=errorPerSample(X, label)
            gradients=backPropagation(X, label, S, weights)
            weights=updateWeights(weights,gradients,alpha)
        
        # Get average error over the epoch
        err[epoch]/=num_examples
    return err, weights

"""
Part 3: Performance Evaluation
"""

'''
Function that calculates prediction of input example
given a list of junction weight matrices
'''
def pred(x_n,weights):
    outputs,_=forwardPropagation(x_n, weights)
    prob=outputs[-1][0]
    return 1 if prob>0.5 else -1

'''
Function that computes the confusion matrix given list
of examples, corresponding labels and weight matrices
'''
def confMatrix(X_train,y_train,w):
    # Initialize training dataset with the bias term
    ones=np.ones((len(X_train), 1))
    X_train=np.hstack((ones, X_train))

    # Find network predictions for all examples
    predictions=[0]*len(y_train)
    for i, example in enumerate(X_train):
        predictions[i]=pred(example, w)

    return confusion_matrix(y_train, predictions)

'''
Function to draw the plot of average network error
over all epochs
'''
def plotErr(e,epochs):
    plt.figure()
    plt.plot(e)
    plt.title('Average Network Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.show()

'''
Function that tests scikit-implementation of MLP classifier
'''
def test_SciKit(X_train, X_test, Y_train, Y_test):
    clf=MLPClassifier(alpha=0.00001, hidden_layer_sizes=(30, 10), random_state=1)
    clf=clf.fit(X_train, Y_train)
    predictions=clf.predict(X_test)
    return confusion_matrix(Y_test, predictions)

'''
Function to plot average network error and compare performace of our 
implementation with scikit MLP classifier
'''
def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is:\n",cM)
    print("Confusion Matrix from Part 1b is:\n",sciKit)

test_Part1()