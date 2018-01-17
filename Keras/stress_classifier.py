from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from stress_data_prep import subjects

from keras import backend as K
from sklearn.model_selection import StratifiedKFold

import time
import numpy as np

def load_all_subjects():
	for i in range(len(subjects)):
		if i is 0:
			X = np.array((subjects[i].hr, subjects[i].bvp, subjects[i].eda))
			print("first pass : X = ", X.shape)
			Y = np.array((subjects[i].binary_output))
		else:
			print("second pass : nparray = ", np.array((subjects[i].hr, subjects[i].bvp, subjects[i].eda)).shape)
			print("second pass : X  =", X.shape)
			X = np.concatenate((X, np.array((subjects[i].hr, subjects[i].bvp, subjects[i].eda))), axis=1)
			print(X.shape)
			Y = np.concatenate((Y, np.array((subjects[i].binary_output))), axis=0)
	X = X.T
	return X, Y

def load_one_subject(n):
	X = np.array((subjects[n].hr, subjects[n].bvp, subjects[n].eda))
	Y = np.array((subjects[n].binary_output))
	X = X.T
	return X, Y


X, Y = load_all_subjects()

#csts
seed = 1337
splits = 5
startingt = time.time()
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model.add(Dense(4, input_dim=3, activation='relu'))
	model.add(Dense(4, activation='relu'))
	model.add(Dense(4, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=10, batch_size=256, verbose=1)
	# Evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("time elapsed : ", time.time() - startingt, " s")
