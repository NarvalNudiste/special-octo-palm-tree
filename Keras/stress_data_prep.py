from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from labelizer import labelize

''' Equalizes array size to match the other one (The larger one gets stripped from his last cells)'''
def resizeArray(a1, a2):
    diff = abs(a1.shape[0] - a2.shape[0])
    if a1.shape[0] < a2.shape[0]:
        a2 = a2[:-diff]
    else:
        a1 = a1[:-diff]
    return a1, a2
# fix random seed for reproducibility
seed = 1337
np.random.seed(seed)

#loading data
eda = np.genfromtxt("data/URIT_1581_A01713_2017_10_31_0946/EDA.csv", delimiter=",")
temp = np.genfromtxt("data/URIT_1581_A01713_2017_10_31_0946/TEMP.csv", delimiter=",")

#resizing arrays
eda, temp = resizeArray(eda, temp)

#stripping the meta informations
eda = eda[2:]
temp = temp[2:]

#plotting
plt.plot(np.linspace(0,eda.shape[0], eda.shape[0]), eda)
plt.plot(np.linspace(0, temp.shape[0], temp.shape[0]), temp)
#plt.show()
data = np.array((eda, temp))
data = data.T

print(data[0])
print(data.shape)

labels = [[0, 0], [30, 1], [60, 0], [90, 1]]
print(labelize(4, labels).shape)
