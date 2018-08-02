# E-commerce-Sales-Predictor


This is a basic machine learning project based on Multivariate Linear Regression based on the cost and gradient  formula.

The dataset we used is a text file having attributes separated by commas.


Here we implement the GRADIENT DESCENT using the help of Python frameworks namely:
      1.Numpy
      2.Pandas
      3.Sklearn(For testing purpose only to split the testing and training data using model_selection attribute) 
      4.Matplotlib (To check the cost function graph with each iteration and to check the linearity of each attribute with the                     expected data)
      
We use the pickle module of python of to save our Revised Theta obtained using the gradient descent locally to file 
with extension .pickle which can be directly used thereafter without training the model again.


The Learning rate we chose was alpha  = 0.0003
You code use anything  lesser than 0.1 so as for the model to work properly


The iterations we decided to be was  100,00,000. You can also chose around 10,000  but we wanted the accuracy to be really 
high. If time constraint for compiling is not a issue then keeping iterations as high as possible is preferred to keep the model from predicting accurately.

