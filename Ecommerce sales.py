"""Mini project to predict the sales of an ecommerce website based on attributes :
1)Avg. session length'
2)Time on mobile app
3)Time on website
4)Membership length"""

#Loading the text data into a CSV file using pandas library

import pandas
filename = 'CustomersData.csv'
names = ['Email','Address','Avatar','Avg. Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent']

#Initialising the dataset as a csv file

data = pandas.read_csv('CustomersData.txt', names=names)

#Dropping the unwanted columns 

data.drop(['Email','Address','Avatar'],axis=1,inplace=True)
print(data.shape)