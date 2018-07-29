####  		Multivariate Linear Regression using Cost function and Gradient Descent

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 




## Data extraction from txt file

dataframe = pd.read_csv('CustomersData.txt')

## Dropping the necessary features
dataframe.drop(['Address','Avatar', 'Email'], 1 , inplace = True)

## Normalizing the dataframe
dataframe = ( dataframe - dataframe.mean() )/ dataframe.std()



### Extracting the features and storing it in Input 
Input = dataframe.iloc[:, 0:4]
## The matrix holding values for X0 whose all elements are 1
ones = np.ones( [Input.shape[0] , 1 ] )
Input = np.concatenate( (ones, Input), axis = 1)

### Extracting the expected output to Output variable
Output = dataframe.iloc[:, 4:5].values


### Initialisation of theta matrix to some random value 
### This theta matrix will go through gradient descent from where the
### the desired value of theta matrix will be obtained from some predefined iterations
theta = np.zeros( [ 1, 5] )
### The parameters for our regression model

# THE LEARNING RATE 
alpha  = 0.01
# THE ITERATIONS FOR GRADIENT DESCENT
iters = 1000



### The Cost function for our model
def cost_function(X, Y , theta  ) :
	Sqr_error = 0
	for i in range(len(Y)):
		Xx = dataframe.iloc[i : i+1 , :]
		Sqr_error += ( (np.dot( theta, Xx.transpose() ) )-Y[i] )**2
	return Sqr_error*(1/(2*len(X)))

# Checking our cost value initially
print(cost_function(Input , Output , theta))



### The Gradient Descent for our model
def gradient_descent(X, Y, theta , alpha , iters):
	for  i in range(iters):

Cost , gradient = gradient_descent(Input , Output ,theta, iters, alpha)
print(gradient)


finalCost = Cost_function(Input, Output , gradient)
print(finalCost)



Hypothesis = np.zeros( [500 ,1 ] )
for i in len(Y):
	Hypothesis = (X[i]@theta.T) - Y[i]
	print(Hypothesis/Y[i])
