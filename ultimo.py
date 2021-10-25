import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from numpy.core.fromnumeric import var
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def train(y, x,y_test,x_test):
    clf = DecisionTreeClassifier()
    clf.fit(x,y)
    #corrector(y,clf.predict(x))
    total=corrector(y_test,clf.predict(x_test))
    return total

def corrector(y,y_pred):
    ans=confusion_matrix(y,y_pred) 
    print(ans)
    print("Porcentaje",(ans[0][0]+ans[1][1])/len(y))
    return (ans[0][0]+ans[1][1])/len(y)

def get_data():
    tipo = {"Female":-1,"Male":1}
    x=[];y=[]

    with open('gender.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i in csv_reader: 
            col=[]
            for j in i: 
                if(j in tipo): col.append(tipo[j])
                else: col.append(float(j))
            y.append(col.pop())
            x.append(np.array(col))
    shuffle_vec_x=[]
    shuffle_vec_y=[]
    s={-1}
    for i in range(len(x)):
        random1=-1
        while(1):
            random1=np.random.randint(0,5001)
            if(random1 not in s):
                s.add(random1)
                break
        shuffle_vec_x.append(x[random1])
        shuffle_vec_y.append(y[random1])

    x=shuffle_vec_x
    y=shuffle_vec_y
    total=0
    block=500
    start=0
    ac=[]
    epocs=[]
    for i in range(10):
        epocs.append(i)
        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]

        for j in range(start):
            x_test.append(x[start])
            y_test.append(y[start])

        if(block==5000): block=5001
        
        for j in range(block):
            x_train.append(x[start])
            y_train.append(y[start])
            start=start+1

        for j in range(start,5001):
            x_test.append(x[j])
            y_test.append(y[j])
        
        tempp=train(y_test,x_test,y_train,x_train)
        total=total+tempp
        ac.append(tempp)
        print(total)
        #print(len(x_test),len(x_train))
    total=total/10
    print("Promedio del accuracy",total)
    plt.plot(epocs, ac,color="green", label="Accuracy")
    plt.legend()
    plt.xlabel("Epoca")
    plt.ylabel("Acurracy")
    plt.show()
np.random.seed(0)
get_data()