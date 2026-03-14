import pandas as pd
data=pd.read_csv("excel1.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)
ypredict=model.predict(xtest)
from sklearn.metrics import r2_score
print(r2_score(ypredict,ytest))