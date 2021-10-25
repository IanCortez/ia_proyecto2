import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


epsilon = 1e-6


def h(w, x, b):
	return (np.dot(np.transpose(w), x) + b)


def s(var_depen, x, b):
	return (1 / (1 + np.exp( -h(var_depen, x, b) )))


def err(x, y, var_depen, b):
	error = 0
	n = len(x)
	for i in range(n):
		error += y[i] * np.log(s(var_depen, x[i], b) + epsilon) + (1-y[i])* np.log(1 - s(var_depen, x[i], b) + epsilon)

	error *= (-1 / n)
	return error


def derivada(x, y, var_depen, b):
	deva = [0 for _ in range(len(var_depen))]
	n = len(x)
	for i in range(len(var_depen)):
		for j in range(n):
			deva[i] += (y[j] - h(var_depen, x[j], b)) * -x[j][i]
		deva[i] /= n
	return deva


def train(y_train, x_train, y_valid, x_valid, y_test, x_test):
	err_train=[]; err_val=[]; est=[]
	var_depen = [np.random.rand() for _ in range(len(x_train[0])) ]
	b = np.random.rand()

	error = err(x_train, y_train, var_depen, b)
	print(error, err(x_valid,y_valid,var_depen,b))
	alfa = 0.001
	epocas = 1000

	for epchs in range(epocas):
		da = derivada(x_train, y_train, var_depen, b)
		for i in range(len(var_depen)):
			var_depen[i] -= da[i]*alfa
		error = err(x_train,y_train,var_depen,b)
		error2 = err(x_valid,y_valid,var_depen,b)
		err_train.append(error)
		err_val.append(error2)
		est.append(epchs)
		epchs += 1

	print("Error train", err(x_train,y_train,var_depen,b))
	print("Error validacion", err(x_valid,y_valid,var_depen,b))
	print("Error testing:", err(x_test, y_test, var_depen, b))

	plt.plot(est, err_train, color="green", label="Training data error")
	plt.plot(est, err_val, color="yellow", label="Validation data error")

	plt.legend()
	plt.xlabel("Epoca")
	plt.ylabel("Error")
	plt.show()


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def main():
	np.random.seed(0)
	file = open("gender_classification.csv")
	reader = csv.reader(file, delimiter=",")

	values = []
	for row in reader:
		values.append(row)
	
	gender = {"Male": 1, "Female": 0}

	y_val = []
	x_val = []
	for i in range(1, len(values)-1):
		col = []
		for j in range(len(values[i])):
			if j == len(values[i])-1:
				y_val.append(gender[values[i][j]])
			else:
				col.append(values[i][j])
		x_val.append(col)
	
	y_val = np.array(y_val, dtype=float)
	x_val = np.array(x_val, dtype=float)

	# Normalizacion de Y
	y_val = NormalizeData(y_val)
	# Normalizacion de X
	x_val = np.transpose(x_val)
	for i in range(len(x_val)): x_val[i] = NormalizeData(x_val[i])
	x_val = np.transpose(x_val)

	
	x_train, x_rem, y_train, y_rem = train_test_split(x_val, y_val, train_size=0.7)
	x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, train_size=0.33)

	train(y_train, x_train, y_valid, x_valid, y_test, x_test)


main()
