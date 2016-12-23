import sklearn

class Classifier(object):
	
	def __init__(self):
		# we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    clf.fit(X, y)

class ClassifierComposite(Classifier):

	def __init__(self):
		self.classifiers = []

	def add(self, classifier):
		self.classifiers.append(classifier)

	def remove(self, classifier):
		self.classifiers.remove(classifier)