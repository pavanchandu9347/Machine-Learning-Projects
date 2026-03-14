import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,4,9,16,25]
plt.plot(x,y)
plt.scatter(x,y,color='red')
plt.xlabel("income")
plt.ylabel("savings")
plt.title("Income and Savings")
plt.bar(x,y,color='yellow')
# plt.pie(x,labels=y)
plt.show()