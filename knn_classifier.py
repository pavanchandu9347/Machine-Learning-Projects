import pandas as pd
data=pd.read_csv("iris.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,ypred)*100)
sw=float(input("Enter sepal width:"))
sl=float(input("Enter sepal length:"))
pw=float(input("Enter petal width:"))
pl=float(input("Enter petal length:"))
print(model.predict([[sw,sl,pw,pl]]))