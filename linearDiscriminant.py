import pandas as pd
data=pd.read_csv("student_depression_dataset.csv")
data = pd.get_dummies(data, drop_first=True)

x=data.drop("Depression",axis=1)
y=data["Depression"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.80,random_state=65)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(xtrain,ytrain)
ypred=lda.predict(xtest)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,ytest)
print(ac*100)


import matplotlib.pyplot as plt
x1=lda.transform(x)
plt.scatter(x1,y, c="blue")
plt.show()