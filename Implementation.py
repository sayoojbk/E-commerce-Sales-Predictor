import pickle
import numpy  as np 
import pandas as pd 
from sklearn import model_selection



pickle_in = open('GradientDescent.pickle', 'rb')
Revised_theta = pickle.load(pickle_in)
#print(Revised_theta)



data_frame = pd.read_csv('CustomersData.txt')



data_frame.drop(['Address','Avatar', 'Email', 'Time on Website'] , 1 , inplace =True)
Input = data_frame.iloc[:, 0:3]
ones = np.ones( [Input.shape[0] , 1 ] )
Input = np.concatenate( (ones, Input), axis = 1)


Output = data_frame.iloc[:, 3:4].values

## Splitting the testing data
## model_selection provides different testing data every time we run it.
X_train , X_test, Y_train, Y_test = model_selection.train_test_split(Input, Output, test_size = 0.2)


######### Checking accuracy of the algorithm

def model_accuracy(X_test, Y_test, Revised_theta):
	accuracy = np.zeros( [Y_test.shape[0] , 1] ) 
	for  i in range( len(Y_test) ):
		X_features = X_test[i]
		Predicted = np.dot( X_features, Revised_theta.transpose() )
		accuracy[i] = 1 - ( abs( Predicted-Y_test[i] )/Y_test[i] )
		accuracy[i] = accuracy[i]* 100
	print( np.mean(accuracy) )

model_accuracy( X_test , Y_test , Revised_theta)
