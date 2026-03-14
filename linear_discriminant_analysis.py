import pandas as pd
data = pd.read_csv("BankNoteAuthentication.csv")

x = data.drop("class",axis=1)
y = data["class"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.80,random_state=65)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(xtrain,ytrain)
ypred=lda.predict(xtest)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,ytest)
print(ac*100)

import matplotlib.pyplot as plt
x1=lda.transform(x)
plt.scatter(x1,y,c=y)
#plt.scatter(x1,y,c="blue")
plt.savefig("LDA.jpeg")
plt.show()