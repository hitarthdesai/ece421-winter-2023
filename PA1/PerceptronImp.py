"""
Part 1: Pocket Algorithm
"""

# Importing necessary modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix

"""
Part 1a: Pocket Algorithm using NumPy and Python
"""

'''
Function used to fit the Perceptron Algorithm to given training examples 
and returns the weight of the best Perceptron Algorithm.
Takes matrix of input examples and corresponding label as input.
'''
def fit_perceptron(X_train, y_train):
    # Defining maximum number of epochs
    numEpochs=5000

    # Appending a column of ones to start of the training examples
    # matrix to handle bias term(element 0 in the weight matrix)
    ones=np.ones((len(X_train), 1))
    X_train=np.hstack((ones, X_train))
    # Initializing an all zero weight matrix
    w=np.zeros((len(X_train[0]), 1))

    for epoch in range(numEpochs):
        # Calculating error over the training examples and ending
        # the fitting process if no examples are misclassified
        error=errorPer(X_train, y_train, w)
        if error==0: return w

        for i, example in enumerate(X_train):
            # Getting the prediction and true label for example
            prediction=pred(example, w)
            truth=y_train[i]
            
            # Updating weight if example was misclassified
            if prediction != truth:
                w=w+truth*np.resize(X_train[i, :], (5,1))
    return w
    
'''
Function to calculate error over given examples, labels and weights.
Return fraction of examples that are mis-classified by given weight matrix.
'''
def errorPer(X_train,y_train,w):
    numMisClassified=0
    for i, example in enumerate(X_train):
        predictedClass=pred(example, w)

        # Incrementing mis-classified count when prediction and truth are unequal
        if predictedClass!=y_train[i]: numMisClassified+=1

    return numMisClassified/X_train.shape[0]
    
'''
Function to obtain the prediction for an example as calculated from a given weight matrix.
Returns +1 for a positive dotProduct and -1 otherwise.
'''
def pred(X_i,w):
    dotProduct=np.dot(X_i, w)
    ans = 1 if dotProduct>0 else -1
    return ans
    
'''
Function that returns confusion matrix obtained by given weight matrix on a set of
training examples and their corresponding labels.
Returns a 2x2 numpy array as
[[True -ve, False-ve]
 [False+ve, True +ve]]
'''
def confMatrix(X_train,y_train,w):
    # Appending a column of ones to start of the training examples
    # matrix to handle bias term(element 0 in the weight matrix)
    ones=np.ones((len(X_train), 1))
    X_train=np.hstack((ones, X_train))
    matrix=np.zeros((2,2))

    for i, example in enumerate(X_train):
        predictedClass=pred(example, w)
        trueClass=y_train[i]

        # Comparing the predicted and true class and incrementing
        # corresponding position in the confusion matrix
        if trueClass==-1:
            if predictedClass==-1: matrix[0][0]+=1
            else: matrix[0][1]+=1
        else:
            if predictedClass==-1: matrix[1][0]+=1
            else: matrix[1][1]+=1

    return np.asarray(matrix, dtype=int)
    
"""
Part 1b: Pocket ALgorithm using scikit-learn
"""

'''
Function that fits a Perceptron Algorithm from scikit-learn to given training
examples and returns confusion matrix for a fitted Perceptron Algorithm
on the given testing data.
'''
def test_SciKit(X_train, X_test, Y_train, Y_test):
    perceptron=Perceptron(max_iter=5000, tol=None)
    weights=perceptron.fit(X_train, Y_train)
    predictions=perceptron.predict(X_test)

    return confusion_matrix(Y_test, predictions)
    
"""
Tests for Part 1
"""

'''
Function to test the NumPy and Scikit implementation of the
Perceptron Algorithm
'''
def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    
test_Part1()
