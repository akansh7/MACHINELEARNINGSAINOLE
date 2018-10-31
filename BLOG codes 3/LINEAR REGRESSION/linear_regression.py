import pandas as pd
#reading the dataset
data=pd.read_csv("data.csv") 

#seperating the independent and dependent variables.
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#splitting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#Calling the LinearRegression class from sklearn.linear_model library
from sklearn.linear_model import LinearRegression
#creating the object of the LinearRegression class
regressor = LinearRegression()
#Fitting the model in training data
regressor.fit(x_train, y_train)

#Predicting 
#prediction is the vector of predictions of the dependent variables.
#predict method of regressor object helps us to find the prediction.
prediction = regressor.predict (x_test) 

#Visualising the Test set result
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, color="Green")
plt.plot(x_test, prediction, color="Blue")
plt.title("x vs y in the test set prediction")
plt.xlabel("x")
plt.ylabel("y")
