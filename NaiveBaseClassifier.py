import pandas as pd
data=pd.read_csv("Diabetes.csv")
x=data.drop("diabetes",axis=1)
y=data["diabetes"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.80,random_state=9)
from sklearn.naive_bayes import GaussianNB
gn=GaussianNB()
gn.fit(xtrain,ytrain)
ypred=gn.predict(xtest)
from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,ytest)
print(ac*100)