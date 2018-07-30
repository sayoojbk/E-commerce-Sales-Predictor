import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math

## Data extraction from txt file

dataframe = pd.read_csv('CustomersData.txt')
#print(dataframe.head())
dataframe.drop(['Address','Avatar', 'Email'], 1 , inplace = True)
#print(dataframe.head())
#dataframe = ( dataframe - dataframe.mean() )/ dataframe.std()
#print(dataframe.head())
### Feature dataset as stored in  Numpy array as float 64


#print(dataframe.shape)

#X = np.array(dataframe.drop(['Yearly Amount Spent'] , 1) , dtype = "float64" )
#print(X.shape)



##for i in  X[0]:
#	print(type(i))


### Extracted labels from dataframe


#Y = np.array(dataframe.drop(['Avg. Session Length','Time on App','Time on Website','Length of Membership'] , 1) , dtype= "float64" )



#print(Y)



####### PLotting the Data for checking the linearity
#for i in range(4):
#	plt.scatter( [A[i] for A in X ] , Y , s = 10)
#	plt.show()

#def gradient_descent(iterations, theta):
#	for  i in range(iterations):
#		pass

Input = dataframe.iloc[:, 0:4]
ones = np.ones( [Input.shape[0] , 1 ] )
Input = np.concatenate( (ones, Input), axis = 1)
#print(X.shape)

Output = dataframe.iloc[:, 4:5].values
#print(Y.shape)
theta = np.zeros( [ 1, 5] )
#print(theta)

alpha  = 0.001
iters = 1000



def cost_function(X, Y, theta):
	tobesummed = np.power( ( (X @ theta.T) - Y ), 2 )
	return np.sum(tobesummed)/( 2*len(X) )

print( cost_function(Input, Output, theta) )
#print( Cost_function(Input, Output, theta) )
def gradient_descent( X, Y , theta, iters , alpha ):
	## This is not necessary this is just to plot the cost function values with each iter
	cost = np.zeros(iters)
	for i in range(iters):
		theta = theta - ( alpha/len(X)*2 )*( np.sum( X*( (X@theta.T) - Y), axis = 0) )
		cost[i] = cost_function(X,Y, theta)
		print(cost[i])

	return theta , cost	

####### The following line will run the gradient descent and give the least cost value

Revised_theta, cost = gradient_descent(Input , Output ,theta, iters, alpha)
print(Revised_theta)
#print(Revised_theta.shape)
#print(Revised_theta)

finalCost = cost_function(Input, Output , Revised_theta)
print(finalCost)



#Hypothesis = np.zeros( [500 ,1 ] )
#for i in len(Y):
#	Hypothesis = (X[i]@theta.T) - Y[i]
#	print(Hypothesis/Y[i])



######### Checking accuracy of the algorithm
#accuracy = 0
#for k in range(10):
#	features = dataframe.iloc[k:k+1 ,0 :4]
	
#	ones = np.ones( [1 , 1 ] )

#	features = np.concatenate( (ones, features), axis = 1)
	#print(features.shape)
	#print(theta.shape)
#	Predicted = np.dot( Revised_theta, features.transpose() )
	#print(Predicted)
	