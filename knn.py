import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def predict(x_split, y_split):
	vecinos = int(input("Nearest neighbours: "))
	k_val = 10

	err_train = []
	err_test = []
	epchs = []

	k_folds = KFold(n_splits=k_val, shuffle=True)
	knn = KNeighborsClassifier(n_neighbors=vecinos, algorithm="kd_tree")
	
	total_err = 0
	i = 1
	
	for train_index, test_index in k_folds.split(x_split):
		# Separtion of the Fold into validation and testing
		x_train, x_test = x_split[train_index], x_split[test_index] 
		y_train, y_test = y_split[train_index], y_split[test_index]

		# Training step
		knn.fit(x_train, y_train)
		err_train.append(1-knn.score(x_train, y_train))
		err_test.append(1-knn.score(x_test, y_test))
		epchs.append(i)
		i += 1
		total_err += 1-knn.score(x_split, y_split)
	
	total_err /= vecinos
	print(f"Error total: {total_err}")

	plt.plot(epchs, err_train, color="red", label="Training data error")
	plt.plot(epchs, err_test, color="green", label="Testing data error")

	plt.legend()
	plt.xlabel("Experiment #")
	plt.ylabel("Error")
	plt.show()


def main():
	np.random.seed(0)
	file = open("gender_classification.csv")
	reader = csv.reader(file, delimiter=",")

	values = []
	for row in reader:
		values.append(row)


	gender = {"Male": 1, "Female": -1}
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

	y_val = NormalizeData(y_val)

	x_val = np.transpose(x_val)
	for i in range(len(x_val)): x_val[i] = NormalizeData(x_val[i])
	x_val = np.transpose(x_val)
	
	predict(x_val, y_val)


main()
