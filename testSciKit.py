from sklearn import datasets, svm
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def createImg():
	im = Image.new("L", (8,8))

	im.show()

	print(digits.data[-1:])
	i = 0
	for x in range(0, 8):
		for y in range(0, 8):
			current_color = im.getpixel( (x,y) )
			newValue = 255 * (1/digits.data[-1:][x][y])
			i = i+1
			im.putpixel( (x,y), 120)
			
	im.show()

def writeClassifierToDisk(clf, path):
	s = pickle.dumps(clf)
	with open(path, "wb") as f:
		f.write(s)

def loadClassifier(path):
	with open(path, "rb") as f:
		str = f.read()
		tempClf = pickle.loads(str)
		return tempClf

def readImage(path):
	arr = []
	for i in range(8):
		arr.append([])
	with Image.open(path) as i:
	
		for y in range(8):
			for x in range(8):
				r, g, b = i.getpixel((x,y))
				arr.append(r)
	
	print(arr)
				#create a tab accordingly
				
				

iris = datasets.load_iris()
digits = datasets.load_digits()

#clf = svm.SVC(gamma = 0.001, C=100)
#lf.fit(digits.data[:-1], digits.target[:-1])

clf = loadClassifier('C:\He-Arc-3\P3\dump.txt')

readImage('C:/He-Arc-3/P3/testImage.png')

#for i in range(0, 5):
#	print("number recognised : ", clf.predict(digits.data[i]))
#	plt.figure(1, figsize=(5, 5))
#	plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
#	plt.show()

#writeClassifierToDisk(clf, 'C:\He-Arc-3\P3\dump.txt')

