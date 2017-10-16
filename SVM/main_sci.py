from sklearn import datasets, svm
from dloader import dataloader
from clfmgr import clfmanager
import warnings

debug_mode = False
learning_mode = True

dumpPath = "saves/clfDump.txt"

def predict(clf, eda, hr):
	print("Prediction -> hr = ", hr, " / eda = ", eda, " || result = ", clf.predict([hr, eda]))
	
def main():
	dl = dataloader('data/HR.csv', 'data/STRESS.csv', 'data/EDA.csv')
	
	mgr = clfmanager()
	clf = svm.SVC(gamma = 0.5, C=100)		
	
	if learning_mode:
		clf.fit(dl.X[:-1], dl.Y[:-1])
		mgr.writeClassifierToDisk(clf, dumpPath)
	else:
		clf = mgr.loadClassifier(dumpPath)
		
	#We now can try to predict stress target value acoording to various inputs
	#EDA values range from 0.0 to 1.000
	#HR values range from 50 to 70
	
	#clf.predict(dl.X[2])
	#print(dl.X[-1:])
	#print(clf.predict(dl.X[-1:]))
	for i in range(len(dl.X)):
		print(i, " - ", clf.predict(dl.X[[i]]))
	#predict(clf, 0.70, 50)

if debug_mode is True:
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		main()
else:
	main()