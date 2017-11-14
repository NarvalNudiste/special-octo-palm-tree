from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

from sklearn.model_selection import StratifiedKFold

import numpy
# fix random seed for reproducibility
seed = 1337
numpy.random.seed(seed)

dataset = numpy.loadtxt("data/pima-indians-diabetes.data", delimiter=",")
# split into input (X) and output (Y) variables


X = dataset[:,0:8]
Y = dataset[:,8]

splits = 10

kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X, Y):
    optimizers = {'adam'}
    for o in optimizers:
        m = Sequential()
        m.add(Dense(32, input_dim=8, activation = 'relu'))
        m.add(Dense(32, activation="relu"))
        m.add(Dense(1, activation="sigmoid"))
        m.compile(loss='binary_crossentropy', optimizer=o, metrics=['accuracy'])
        m.fit(X[train], Y[train], epochs=1000, batch_size=128, verbose=0)
        scores = m.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
