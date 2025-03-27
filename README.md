# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import Required Libraries

2.Load Dataset

3.Preprocess Data 

4.Split Data into Training & Testing Sets 

## Program:
```
Developed by:priyanka R
RegisterNumber:212223220081
```
~~~
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
~~~

![image](https://github.com/user-attachments/assets/ac75b79b-c191-4408-9c42-f15daad905f4)
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

![image](https://github.com/user-attachments/assets/fc4866ed-4abc-413c-9f26-68aad551c3a0)
```
data1.isnull().sum()
```

![image](https://github.com/user-attachments/assets/151b57e5-49be-4d46-9e2f-c521b81c8ecc)

data1.duplicated().sum()
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
![image](https://github.com/user-attachments/assets/1b5331a9-76bb-4acd-b331-4446144522c5)
```
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/da97e4ce-15c1-416e-b634-f0c0a9249549)
~~~
y=data1["status"]
y
~~~

![image](https://github.com/user-attachments/assets/2c44d758-96c1-4d1a-a5e3-374d15c27f1e)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```

![image](https://github.com/user-attachments/assets/81c18586-79c4-420b-8ac9-432f2443fd34)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/f5e8e062-1638-4138-8346-01ce097099ee)
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/b5842f5d-6a4b-4785-a631-e44a36c68247)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/46383d70-b336-436d-a207-1c942092adbf)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
