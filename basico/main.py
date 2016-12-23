import numpy as np
import pandas as pd
import ipdb
import sklearn.neighbors
import sklearn.metrics

def print_header():
	print()
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	print(">>>>>  DESAFIO INTELIVIX  >>>>>")
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	print()
	print(">> Iris data set:")
	print()

def print_exp_progress(atual, maximum):
	print('Running experiment:',atual, 'of', maximum)

def get_dataframe():
	import io
	import requests

	# Collect data
	iris_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	iris_csv = requests.get(iris_url).content
	return pd.read_csv(io.StringIO(iris_csv.decode('utf-8')), header=None)

def main():
	
	iris_df = get_dataframe()
	labels = iris_df[4].unique()
	
	for i, label in enumerate(labels):
		text = 'Code: {1} | Label: {0}'.format(label,i)
		print(text)

	print()

	# Store confusion matrix experiments results
	nof_exps = 50
	conf_matrix_exps = np.zeros([nof_exps,3,3])

	# Teorema do limite central
	for i in range(nof_exps):

		print_exp_progress(i+1,nof_exps)

		#Shuffle data for each class
		USAR ISSO -> http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

		def data(array):
			return array[:,:4]

		def target(array):
			return array[:,4]

		# Split data
		train = np.array(iris_df[msk])
		test = np.array(iris_df[~msk])

		print()
		print('Training set size:', train.shape[0])
		print('Testin set size:', test.shape[0])
		print()

		# Training
		nof_neighbors = 8
		clf = sklearn.neighbors.KNeighborsClassifier(nof_neighbors, weights='distance')
		clf.fit(data(train),target(train))
		
		# Testing
		answers = clf.predict(data(test))

		conf_matrix_exps[i] = np.array(sklearn.metrics.confusion_matrix(target(test), answers))

	# Result analysis
	conf_matrix_mean = np.zeros([len(labels),len(labels)])
	conf_matrix_std = np.zeros([len(labels),len(labels)])
	np.mean(conf_matrix_exps, axis=0, out = conf_matrix_mean)
	np.std(conf_matrix_exps, axis=0, out = conf_matrix_std)

	print()
	print('Confusion matrix (mean values for ',nof_exps,' experiments):')
	print(conf_matrix_mean)
	print()

	print('Confusion matrix (std values for ',nof_exps,' experiments):')
	print(conf_matrix_std)
	print()

if __name__ == '__main__':
	print_header()
	main()


