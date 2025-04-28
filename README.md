# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmLoad the Dataset

1.Load Dataset: Load the Placement_Data.csv dataset using pandas.read_csv() to create a DataFrame.

2.Remove Unnecessary Columns: Drop irrelevant columns (such as sl_no and salary) from the dataset.

3.Check for Missing Values and Duplicates: Check for any missing values (isnull().sum()) and duplicate rows (duplicated().sum()) in the data.

4.Label Encoding: Use LabelEncoder to convert categorical columns (like gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation, status) into numerical values.

5.Feature Selection: Separate the dataset into features (X) and the target variable (y), where y is the status (placement status) and X includes the other columns.

6.Split the Data: Split the dataset into training and testing sets using train_test_split(), with 80% of the data for training and 20% for testing.

7.Train the Model: Create and train a logistic regression model using LogisticRegression() and the training data (X_train, y_train).

8.Predict on Test Data: Use the trained model to predict the placement status (y_pred) for the test data (X_test).

9.Evaluate the Model: Calculate the accuracy using accuracy_score(), generate the confusion matrix using confusion_matrix(), and create a classification report using classification_report().

10.Make Predictions on New Data: Use the trained model to predict the placement status of new student data (e.g., lr.predict([[...]])).

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Divya R
RegisterNumber:  212222040040
*/
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver= "liblinear")
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)
print("Name: Divya R")
print("Reg no : 212222040040")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## data.head()
![image](https://github.com/user-attachments/assets/9a72d67e-d57a-43c9-8f4e-eec25d1f4e6d)
## data1.head()
![image](https://github.com/user-attachments/assets/8eb5505d-4158-4400-88b4-5477a2309f5d)
## isnull()
![image](https://github.com/user-attachments/assets/0e1c4c75-1883-4133-adc8-5ca8944e06c1)
## duplicated()
![image](https://github.com/user-attachments/assets/570169e2-03f8-4113-a3f1-4c7089dfc07e)
## data1
![image](https://github.com/user-attachments/assets/567238f7-1fde-47ad-8c79-cea8f857fd8c)
## X
![image](https://github.com/user-attachments/assets/fea8cee6-4e28-4a18-b8e9-c2a99005d6db)
## y
![image](https://github.com/user-attachments/assets/7fc199f9-cae5-405e-b6e4-3c259c8d4d5e)
## y_pred
![image](https://github.com/user-attachments/assets/c0bced6d-257a-4810-aeb7-798844d93bad)
## confusion matrix
![image](https://github.com/user-attachments/assets/4cf95e7a-9a46-446a-8e56-1626808b25dd)
## classification report
![image](https://github.com/user-attachments/assets/20d1f245-8899-47c0-9845-bed4ab20201d)

## prediction
![image](https://github.com/user-attachments/assets/72803270-8e5f-4e5b-a17f-682fb66075d3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
