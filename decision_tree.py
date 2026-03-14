import pandas as pd
data=pd.read_csv("dataset_traffic_accident_prediction1.csv")
#print(data)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Weather']=le.fit_transform(data['Weather'])
data['Weather']=data['Weather'].fillna(data['Weather'].mean())
data['Road_Type']=le.fit_transform(data['Road_Type'])
data['Road_Type']=data['Road_Type'].fillna(data['Road_Type'].mean())
data['Time_of_Day']=le.fit_transform(data['Time_of_Day'])
data['Time_of_Day']=data['Time_of_Day'].fillna(data['Time_of_Day'].mean())
data['Traffic_Density']=data['Traffic_Density'].fillna(data['Traffic_Density'].mean())
data['Speed_Limit']=data['Speed_Limit'].fillna(data['Speed_Limit'].mean())
data['Number_of_Vehicles']=data['Number_of_Vehicles'].fillna(data['Number_of_Vehicles'].mean())
data['Driver_Alcohol']=data['Driver_Alcohol'].fillna(data['Driver_Alcohol'].mean())
data['Accident_Severity']=le.fit_transform(data['Accident_Severity'])
data['Accident_Severity']=data['Accident_Severity'].fillna(data['Accident_Severity'].mean())
data['Road_Condition']=le.fit_transform(data['Road_Condition'])
data['Road_Condition']=data['Road_Condition'].fillna(data['Road_Condition'].mean())
data['Vehicle_Type']=le.fit_transform(data['Vehicle_Type'])
data['Vehicle_Type']=data['Vehicle_Type'].fillna(data['Vehicle_Type'].mean())
data['Driver_Age']=data['Driver_Age'].fillna(data['Driver_Age'].mean())
data['Driver_Experience']=data['Driver_Experience'].fillna(data['Driver_Experience'].mean())
data['Road_Light_Condition']=le.fit_transform(data['Road_Light_Condition'])
data['Road_Light_Condition']=data['Road_Light_Condition'].fillna(data['Road_Light_Condition'].mean())
data['Accident']=le.fit_transform(data['Accident'])
data['Accident']=data['Accident'].fillna(data['Accident'].mean())
print(data)

x=data.iloc[:,:-1].values # ==> denotion considering all rows and all columns except last column
y=data.iloc[:,-1].values  #==> only the selection of last column and all rows ==> denotion
#x=data.iloc[0:840,0:13].values
#y=data.iloc[0:840,13].valueszz
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
#comparing ypred and ytest
from sklearn.metrics import accuracy_score
#classification -> accuracy_score
print(accuracy_score(ypred,ytest))
