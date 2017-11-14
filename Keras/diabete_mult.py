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

X_train = X[:576]
X_test = X[576:]

Y_train = Y[:576]
Y_test = Y[576:]

#callbacks = list()
#tbCallback = TensorBoard(log_dir='logs', histogram_freq=0,
#          write_graph=True, write_images=True)
#callbacks.append(tbCallback)

# for the record https://machinelearningmastery.com/improve-deep-learning-performance/ "No one knows. No one. Don’t ask."

optimizers = {'adam'}
nodesNb = {2, 4}
with open("logs/testEval.txt", 'w') as f:
    tw = ""
    for i in nodesNb:
        for o in optimizers:
            for k in range(4,8):
                print("{:d} Hidden layers, {:d} nodes (Dense) : \n\n".format(k,i))
                tw += "{:d} Hidden layers, {:d} nodes (Dense) : \n\n".format(k,i)
                tw += "| Optimizer         | Accuracy         |\n|------------------|------------------:|\n"
                m = Sequential()
                for elem in range(k):
                    m.add(Dense(i, input_dim=8, activation = 'relu'))
                m.add(Dense(1, activation="sigmoid"))
                m.compile(loss='binary_crossentropy', optimizer=o, metrics=['accuracy'])
                m.fit(X_train, Y_train, epochs=1000, batch_size=128, verbose=1)
                print("{:s}, {:d} nodes done".format(o, i))
                scores = m.evaluate(X_test, Y_test)
                print("\nTesting scores : \n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
                tw += "| {:s}              | {:f}%                  |\n".format(o, scores[1]*100)
            tw += "\n\n"
f.write(tw)


#Wsave = m.get_weights()
#m.fit(X_train, Y_train, epochs=1000, batch_size=128, verbose=1)
#m.set_weights(Wsave)

# evaluate the model
scores = m.evaluate(X_test, Y_test)
print("\nTesting scores : \n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
