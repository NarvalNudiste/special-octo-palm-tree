import numpy as np
from numpy import genfromtxt

class dataloader():
	'''Class used to load different .cvs files and to build our vlues / target array
	'''
	def __init__(self, HR_path, STRESS_path, EDA_path):
	#the __init__ magic function loads the different files and stores them to the X (values) and self.Y (target values) member arrays
	
		#Extracting csv data
		self.hr_data = np.genfromtxt(HR_path, delimiter=' ')
		self.stress_data = np.genfromtxt(STRESS_path, delimiter=' ')
		self.eda_data = np.genfromtxt(EDA_path, delimiter=' ')
		

		#We declare a 2-dimensional array to store values
		self.X = np.empty([len(self.stress_data), 2])
		for i, (hr, eda) in enumerate(zip(self.hr_data, self.eda_data)):
			self.X[i] = (hr, eda)
		#Merci grand sachem
		
		#TODO : fix 1d scikit deprecation issue
		self.Y = np.array(self.stress_data)