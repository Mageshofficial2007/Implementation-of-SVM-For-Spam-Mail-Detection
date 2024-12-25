# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

8.End the Program

## Program:

```python
import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result
```

```python
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()
data.info()
data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc
```

Program to implement the SVM For Spam Mail Detection..

Developed by: MAGESH BOOPATHI.M

RegisterNumber:  24900855

## Output:
RESULT:

![Screenshot 2024-12-25 212226](https://github.com/user-attachments/assets/db8c61eb-7e20-4135-a265-7a6f9e24b1eb)

HEAD:

![Screenshot 2024-12-25 212240](https://github.com/user-attachments/assets/3f02fbfa-7102-49af-9a7e-b3abc554c94a)

COUNTVECTORIZER:

![Screenshot 2024-12-25 212256](https://github.com/user-attachments/assets/dc29d5d6-7187-4e9d-bbe1-3807efb0ce0b)

OUTPUT:

![Screenshot 2024-12-25 212305](https://github.com/user-attachments/assets/957fd48c-22b3-4bcc-8955-c2314bd8bfa2)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
