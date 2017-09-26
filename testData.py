from sklearn import datasets, svm
import numpy as np
from numpy import genfromtxt
import warnings

debug_mode = False

def predict(clf, eda, hr):
	print("Prediction -> hr = ", hr, " / eda = ", eda, " || result = ", clf.predict([hr, eda]))
	
def main():
	dl = dataLoader('test_data/HR.csv', 'test_data/STRESS.csv', 'test_data/EDA.csv')

	clf = svm.SVC(gamma = 0.001, C=100)
	clf.fit(dl.X[:-1], dl.Y[:-1])

	#We now can try to predict stress target value acoording to various inputs
	#EDA values range from 0.0 to 1.000
	#HR values range from 50 to 70
	
	predict(clf, 0.16, 50)
	


class dataLoader():
	'''Tiny class used to load different .cvs files and to build our vlues / target array
	'''
	def __init__(self, HR_path, STRESS_path, EDA_path):
	#the __init__ magic function loads the different files and stores them to the X (values) and self.Y (target values) member arrays
	
		#opening various files, extracting csv data
		with open(HR_path):
			self.hr_data = np.genfromtxt(HR_path, delimiter=' ')
		with open(STRESS_path):
			self.stress_data = np.genfromtxt(STRESS_path, delimiter=' ')
		with open(EDA_path):
			self.eda_data = np.genfromtxt(EDA_path, delimiter=' ')
		

		#We declare a 2-dimensional array to store values
		self.X = np.empty([len(self.stress_data), 2])
		
		for i in range(len(self.stress_data)):
			self.X[i] = ([self.hr_data[i], self.eda_data[i]])
		
		#TODO : fix 1d scikit deprecation issue
		self.Y = np.array(self.stress_data)
		

if debug_mode is True:
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		main()
else:
	main()