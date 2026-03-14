import pandas as pd
file=pd.read_csv('Student_Performance.csv')
x=file.iloc[:,:-1].values
y=file.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5,random_state=0)
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor(n_neighbors=10)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(ytest,ypred))
H_S=float(input("Enter the no.of hours studied:"))
P_S=float(input("Enter the previous score:"))
E_A=int(input("Enter the number of extracurricular actiivities:"))
S_H=float(input("Enter the no. of sleep hours:"))
S_QP_P=int(input("Enter no.of sample question paper practised:"))
print(model.predict([[H_S,P_S,E_A,S_H,S_QP_P]]))