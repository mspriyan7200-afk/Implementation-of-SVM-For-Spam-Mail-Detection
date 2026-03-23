# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYAN M
RegisterNumber: 212225040320 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv("C:/Users/acer/Downloads/spam (1).csv", encoding='latin-1')


data = data[['v1', 'v2']]
data.columns = ['label', 'message']


data['label'] = data['label'].map({'ham':0, 'spam':1})


X = data['message']
y = data['label']


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
<img width="628" height="55" alt="Screenshot 2026-03-23 183951" src="https://github.com/user-attachments/assets/b3e0ceac-bb45-462f-9fb3-ec263d6fc338" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
