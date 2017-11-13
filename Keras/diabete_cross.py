from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
import numpy
# fix random seed for reproducibility
numpy.random.seed(1337)

dataset = numpy.loadtxt("data/pima-indians-diabetes.data", delimiter=",")
# split into input (X) and output (Y) variables



X = dataset[:,0:8]
Y = dataset[:,8]

k = 4
slice_size = int(X.shape[0]/k)

splittedX = numpy.split(X, k)
splittedY = numpy.split(Y, k)


x_train = numpy.empty_like(splittedX[0])

for i in range(k-1):
    x_train = numpy.empty_like(splittedX[0])
    y_train = splittedY[i]
    indices = list(range(0, k))
    indices.remove(i)
    for j in indices:
        print("adding index{:d}".format(j))
        numpy.append(splittedX[j],x_train)
    print(x_train.shape)



#X_train = X[:576]
#X_test = X[576:]

#Y_train = Y[:576]
#Y_test = Y[576:]

#callbacks = list()
#tbCallback = TensorBoard(log_dir='logs', histogram_freq=0,
#          write_graph=True, write_images=True)
#callbacks.append(tbCallback)

m = Sequential()
m.add(Dense(8, input_dim=8, activation = 'relu'))
m.add(Dropout(8))
m.add(Dense(8, input_dim=8, activation = 'relu'))

m.add(Dense(1, activation="sigmoid"))
m.summary()

m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

input("- Model compiled, press enter to train ")
#m.fit(X, Y, epochs=500, batch_size=128, callbacks = callbacks)
m.fit(X_train, Y_train, epochs=500, batch_size=128, verbose=1)


#for i in range(k):
    #x_train = X[]


# evaluate the model
#scores = m.evaluate(X, Y)
scoresT = m.evaluate(X_train, Y_train)
print("Training scores : \n%s: %.2f%%" % (m.metrics_names[1], scoresT[1]*100))
scores = m.evaluate(X_test, Y_test)
print("Testing scores : \n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
