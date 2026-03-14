import pandas as pd
data=pd.read_csv("car_evaluation.csv")

columns=["1","2","3","4","5","6","target"]
data.columns=columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['1']=le.fit_transform(data['1'])
data['2']=le.fit_transform(data['2'])
data['3']=le.fit_transform(data['3'])
data['4']=le.fit_transform(data['4'])
data['5']=le.fit_transform(data['5'])
data['6']=le.fit_transform(data['6'])
x=data.drop("target",axis=1)
y=data["target"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=2)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytest))