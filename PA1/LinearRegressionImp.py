"""
Part 2: Linear Regression
"""

# Importing necessary modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

"""
Part 2a: Linear Regression using scikit-learn
"""

'''
Function to compute the w parameter of the best-fitting plane
for given dataset
'''
def fit_LinRegr(X_train, y_train):
    numExamples, numFeatures=X_train.shape

    # Appending a column of ones at the beginning of the training examples
    # to account for the bias term
    ones=np.ones((numExamples, 1))
    X_train=np.hstack((X_train, ones))

    # Using linalg.pinv to handle case where determinant of matrix given as 
    # parameter is zero
    w=np.dot((np.linalg.pinv(np.dot(X_train.T,X_train))), np.dot(X_train.T,y_train))
    return w.T

'''
Function to calculate the mean square error between truth values and prediction
for all examples in the training dataset
'''
def mse(X_train,y_train,w):
    numExamples, numFeatures=X_train.shape

    # Appending a column of ones at the beginning of the training examples
    # to account for the bias term
    ones=np.ones((numExamples, 1))
    X_train=np.hstack((X_train, ones))

    sumOfSquaredDifference=0
    for i, example in enumerate(X_train):
        prediction=pred(example, w)
        truth=y_train[i]

        # Calculating the difference between prediction and its square
        difference=truth-prediction
        sumOfSquaredDifference+=difference*difference
    meanSquaredError=sumOfSquaredDifference/(i+1)

    return meanSquaredError

'''
Function to calculate prediction of a datapoint for given weight vector w
'''
def pred(X_train,w):
    w=np.transpose(w)
    return np.matmul(X_train, w)

"""
Part 2b: Linear Regression using NumPy and Python
"""

'''
Function that fits a Linear Regression Algorithm from scikit-learn to given
training examples and returns mean-squared error for a fitted LinReg Algorithm
on the given testing data.
'''
def test_SciKit(X_train, X_test, Y_train, Y_test):
    linReg=linear_model.LinearRegression()
    linReg=linReg.fit(X_train, Y_train)
    predictions=linReg.predict(X_test)
    meanSquaredError=mean_squared_error(Y_test, predictions)

    return meanSquaredError

"""
Tests for Part 2
"""

'''
Function that tests if the Numpy implementation works for singular matrices
'''
def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

'''
Function that tests the NumPy and scikit implementation of the Linear
Regression Algorithm
'''
def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
     
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()

"""
Results: Our implementation performs very identical to the scikit-learn
implementation with the first difference in the mean squared error being
observed at the fifth/sixth decimal place.
"""

