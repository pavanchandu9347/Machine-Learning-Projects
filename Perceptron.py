import pandas as pd
data = pd.read_csv("BankNoteAuthentication.csv")

x=data.drop("class",axis=1)
y=data["class"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.80,random_state=65)

from sklearn.linear_model import Perceptron
pc=Perceptron()
pc.fit(xtrain,ytrain)
ypred=pc.predict(xtest)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,ytest)
print(ac*100)

import matplotlib.pyplot as plt
x1=data["variance"]
x2=data["entropy"]
plt.scatter(x1,x2,c=y)
plt.savefig("Perceptron.jpeg")
plt.show()
