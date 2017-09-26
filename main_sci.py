from sklearn import datasets, svm
from dloader import dataloader
import warnings
debug_mode = False

def predict(clf, eda, hr):
	print("Prediction -> hr = ", hr, " / eda = ", eda, " || result = ", clf.predict([hr, eda]))
	
def main():
	dl = dataloader('test_data/HR.csv', 'test_data/STRESS.csv', 'test_data/EDA.csv')

	clf = svm.SVC(gamma = 0.001, C=100)
	clf.fit(dl.X[:-1], dl.Y[:-1])

	#We now can try to predict stress target value acoording to various inputs
	#EDA values range from 0.0 to 1.000
	#HR values range from 50 to 70
	
	predict(clf, 0.60, 60)
	predict(clf, 0.30, 40)

if debug_mode is True:
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		main()
else:
	main()