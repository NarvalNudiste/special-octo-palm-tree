import pickle

class clfmanager():
	'''Class used to write svm classifiers to disk with pickle
	'''
	def writeClassifierToDisk(self, clf, path):
		s = pickle.dumps(clf)
		with open(path, "wb") as f:
			f.write(s)

	def loadClassifier(self, path):
		with open(path, "rb") as f:
			str = f.read()
			tempClf = pickle.loads(str)
			return tempClf