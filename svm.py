import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy.core.fromnumeric import var
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def h(w, x, b):
    return (np.dot(np.transpose(w), x)) + b

def error(y,w, x, b, c):
  w_range=np.dot(np.transpose(w),w)/2
  err=0
  for i in range(len(x)): err += max(0, 1- y[i] * h(w, x[i], b))
  return w_range + c*err

def train(y, x,y_test,x_test,y_valid,x_valid):

    b = np.random.rand()
    w = [np.random.rand() for _ in range(len(x[0]))]

    err_list = []
    err_list_val = []
    err_list_test = []

    epc_list = []

    c= 1
    alfa = 0.001
    epocas = 1000
    
    print(w)

    for epoc in range(epocas):
        
        p = np.random.randint(0, len(x))
        x_val = x[p]
        y_val = y[p]
        err_list.append(error(y,w,x,b,c))
        err_list_test.append(error(y_test,w,x_test,b,c))
        err_list_val.append(error(y_valid,w,x_valid,b,c))
        epc_list.append(epoc)

        if(epoc%100==0): print(error(y,w,x,b,c))

        for i in range(len(w)):
            temp=h(w, x_val, b)*y_val
            if(temp >= 1): w[i] = w[i]-alfa*w[i]
            else:  
                w[i] = w[i]-alfa*(w[i] - x_val[i]*y_val*c)
                b = b+alfa*c*y_val

    print(w)

    plt.plot(epc_list, err_list,color="green", label="Training data error")
    plt.plot(epc_list, err_list_val,color="red", label="Validation data error")
    plt.plot(epc_list, err_list_test,color="yellow", label="Test data error")
    plt.legend()
    plt.xlabel("Epoca")
    plt.ylabel("Error")
    plt.show()
    
    corrector(w,b,x,y)
    corrector(w,b,x_test,y_test)
    corrector(w,b,x_valid,y_valid)

def corrector(w,b,x,y):
    y_pred=[]
    for i in range(len(x)):
        pred=h(w,x[i],b)
        if(pred>=0): y_pred.append(1)
        else: y_pred.append(-1)
    
    ans=confusion_matrix(y,y_pred) 
    print(ans)
    print("Porcentaje",(ans[0][0]+ans[1][1])/len(x))


def Norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

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
    
    x=np.transpose(x)
    for i in range(len(x)): x[i]=Norm(x[i])
    x=np.transpose(x)
    for i in range(5): print(x[i])
    X_train, X_rem, Y_train, Y_rem = train_test_split(x, y, train_size=0.7)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_rem,Y_rem, test_size=0.666223)
    print(len(X_train),len(X_test),len(X_valid))
    train(Y_train, X_train,Y_test,X_test,Y_valid,X_valid)

np.random.seed(0)
get_data()