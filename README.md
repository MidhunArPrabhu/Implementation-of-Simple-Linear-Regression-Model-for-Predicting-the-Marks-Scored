# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. \

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber:  212222240066
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![ML21](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/7abbb39d-c422-48df-b3f5-a4c91fc66835)

df.tail()

![ML22](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/b5912227-43e3-4902-a100-43c5668b97d7)

Array value of X

![ML23](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/2e123c76-04cb-4f9f-b0e0-3705ce2f3a1a)

Array value of Y

![ML24](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/b9768567-d931-4794-96d0-ed72e3999811)
Values of Y prediction

![ML25](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/09bd5d61-e6cd-40b4-95f3-ce95f2a29e39)
Array values of Y test

![ML26](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/6ab94760-076b-4cc1-bb80-c034335fe004)

Training Set Graph

![ML27](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/b80b81f1-f5f2-45d7-87f9-8949f2e609ce)

Test Set Graph

![ML28](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/4847bedf-f7ac-4f97-a829-1ba23f0ec47e)

Values of MSE, MAE and RMSE

![ML29](https://github.com/MidhunArPrabhu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118054670/f1ab8c2b-7c66-4407-90a1-ec1399ba06a9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
