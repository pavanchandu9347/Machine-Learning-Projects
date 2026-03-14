import pandas as pd
data = pd.read_csv("heart.csv")
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(xtrain,ytrain)
ypred = classifier.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)