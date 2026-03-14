import pandas as pd
data=pd.read_csv("diabetes_prediction_india.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Age']=le.fit_transform(data['Age'])
data['Gender']=le.fit_transform(data['Gender'])
data['BMI']=le.fit_transform(data['BMI'])
data['Family_History']=le.fit_transform(data['Family_History'])
data['Physical_Activity']=le.fit_transform(data['Physical_Activity'])
data['Diet_Type']=le.fit_transform(data['Diet_Type'])
data['Smoking_Status']=le.fit_transform(data['Smoking_Status'])
data['Alcohol_Intake']=le.fit_transform(data['Alcohol_Intake'])
data['Stress_Level']=le.fit_transform(data['Stress_Level'])
data['Hypertension']=le.fit_transform(data['Hypertension'])
data['Cholesterol_Level']=le.fit_transform(data['Cholesterol_Level'])
data['Fasting_Blood_Sugar']=le.fit_transform(data['Fasting_Blood_Sugar'])
data['Postprandial_Blood_Sugar']=le.fit_transform(data['Postprandial_Blood_Sugar'])
data['HBA1C']=le.fit_transform(data['HBA1C'])
data['Heart_Rate']=le.fit_transform(data['Heart_Rate'])
data['Waist_Hip_Ratio']=le.fit_transform(data['Waist_Hip_Ratio'])
data['Urban_Rural']=le.fit_transform(data['Urban_Rural'])
data['Health_Insurance']=le.fit_transform(data['Health_Insurance'])
data['Regular_Checkups']=le.fit_transform(data['Regular_Checkups'])
data['Medication_For_Chronic_Conditions']=le.fit_transform(data['Medication_For_Chronic_Conditions'])
data['Pregnancies']=le.fit_transform(data['Pregnancies'])
data['Polycystic_Ovary_Syndrome']=le.fit_transform(data['Polycystic_Ovary_Syndrome'])
data['Glucose_Tolerance_Test_Result']=le.fit_transform(data['Glucose_Tolerance_Test_Result'])
data['Vitamin_D_Level']=le.fit_transform(data['Vitamin_D_Level'])
data['C_Protein_Level']=le.fit_transform(data['C_Protein_Level'])
data['Thyroid_Condition']=le.fit_transform(data['Thyroid_Condition'])
data['Diabetes_Status']=le.fit_transform(data['Diabetes_Status'])
# print(data)
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytest)*100)

