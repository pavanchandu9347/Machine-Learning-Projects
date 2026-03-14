import pandas as pd
data=pd.read_csv("iris.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.neighbors import RadiusNeighborsClassifier
model=RadiusNeighborsClassifier(radius=3)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,ypred)*100)