import pandas as pd 
file=pd.read_csv("housing.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
file['ocean_proximity']=le.fit_transform(file['ocean_proximity'])
file=file.dropna()    
file.fillna(file.mean(numeric_only=True),inplace=True)
x=file.drop("median_house_value",axis=1)
y=file["median_house_value"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=73)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
from sklearn.metrics import r2_score
r=r2_score(ytest,ypred)
print(r*100)

import matplotlib.pyplot as plt
plt.scatter(ytest, ypred, c=ypred)
plt.savefig("random.png")
plt.show()
