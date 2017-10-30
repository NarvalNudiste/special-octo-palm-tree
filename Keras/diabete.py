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

callbacks = list()
tbCallback = TensorBoard(log_dir='logs', histogram_freq=0,
          write_graph=True, write_images=True)
callbacks.append(tbCallback)

m = Sequential()
m.add(Dense(16, input_dim=8, activation = 'relu'))
m.add(Dropout(16))
for i in range(8):
  m.add(Dense(16, activation = 'relu'))
  m.add(Dropout(16))
m.add(Dense(1, activation="sigmoid"))
m.summary()

m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

input("- Model compiled, press enter to train ")
m.fit(X, Y, epochs=10000, batch_size=128, callbacks = callbacks)


# evaluate the model
scores = m.evaluate(X, Y)
print("\n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
