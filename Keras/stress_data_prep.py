from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from labelizer import labelize
import time
import json

path_a = "data/URIT_1581_A01212B_2017_10_31_0946/"
path_b = "data/URIT_1581-A01713-2017_10_31_0949/"
#TODO égaliser les tableaux - done
#TODO Faire une interpolation sur les fréq. d'échantillonnages différentes - done
#TODO ptêtre partir depuis le 64hz en fait - done
#Séprarer en activités aussi, plus tard

class Person:
    def __init__(self, start, stop, overall_health, energetic, overall_stress, stressed_past_24h, sleep_quality_past_24h,  sleep_quality_past_month, data, id):
        self.timestamps = (start, stop)
        self.overall_health = overall_health
        self.energetic = energetic
        self.overall_stress = overall_stress
        self.stressed_past_24h = stressed_past_24h
        self.sleep_quality_past_24h = sleep_quality_past_24h
        self.sleep_quality_past_month = sleep_quality_past_month
        self.data = data
        self.id = id

''' Equalizes array size to match the other one (The larger one gets stripped from his last cells)'''
def resize_ary(a1, a2):
    diff = abs(a1.shape[0] - a2.shape[0])
    if a1.shape[0] < a2.shape[0]:
        a2 = a2[:-diff]
    else:
        a1 = a1[:-diff]
    return a1, a2

''' creates a new array with intermediates values (linear interpolation) to match target freq ary '''
def reshape_array_freq(baseFreq, freq, ary):
    if baseFreq == freq:
        return ary
    else:
        newArray = np.empty(ary.shape[0] * int(freq/baseFreq))
        for i in range(len(ary)-1):
            delta = ary[i] - ary[i+1]/freq
            for c in range(int(freq)):
                newArray[i+c] = ary[i] + delta*c
        return newArray

'''maybe not needed'''
def stripDowntime(array, timestamps):
    offset = 0
    for ts in timestamps:
        print(ts[0], ts[1])
        if ts[1] is -1:
            array = np.delete(array, np.s_[ts[0]:array.shape[0]], axis = 0)
        else:
            array = np.delete(array, np.s_[ts[0] - offset :ts[1] - offset], axis = 0)
            offset += abs(ts[1]-ts[0])
    return array

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


data_ary = list()

#loading data
#TODO find a data structure for ACC and IBI
eda_a = np.genfromtxt(path_a + "EDA.csv", delimiter=",")
temp_a = np.genfromtxt(path_a + "TEMP.csv", delimiter=",")
bvp_a = np.genfromtxt(path_a + "BVP.csv", delimiter=",")
hr_a = np.genfromtxt(path_a + "HR.csv", delimiter=",")

eda_b = np.genfromtxt(path_b + "EDA.csv", delimiter=",")
temp_b = np.genfromtxt(path_b + "TEMP.csv", delimiter=",")
bvp_b = np.genfromtxt(path_b + "BVP.csv", delimiter=",")
hr_b = np.genfromtxt(path_b + "HR.csv", delimiter=",")

data_ary.append(eda_a)
data_ary.append(temp_a)
data_ary.append(bvp_a)
data_ary.append(hr_a)
data_ary.append(eda_b)
data_ary.append(temp_b)
data_ary.append(bvp_b)
data_ary.append(hr_b)

print("eda shape : ", eda_a.shape)

#eda_b = np.delete(eda_b, np.s_[5000:8500], axis = 0)
#temp_b = np.delete(temp_b, np.s_[5000:8500], axis = 0)

timestamps1 = [(0, 1500), (5310, 8150), (11425, 12493), (16170, 17360), (20800, 21813), (25316, -1)]

#b1 = batch(0, 1500, 4, 4, 1, id = 0)
#b2 = batch(5310, 8150, 5, 4, 1, id = 1)
#b3 = batch(11425, 12493, 4, 4, 3, id = 2)
#b4 = batch(16170, 17360, 0, 0, 0, id = 3)
#b5 = batch(20800, 21813, 0, 0, 0, id = 4)


#eda_a = stripDowntime(eda_a, timestamps1)
#temp_a = stripDowntime(temp_a, timestamps1)
#eda_b = stripDowntime(eda_b, timestamps2)
#temp_b = stripDowntime(temp_b, timestamps2)

#resizing arrays
#eda_a, temp_a = resize_ary(eda_a, temp_a)
#eda_b, temp_b = resize_ary(eda_b, temp_b)

#plotting
#plt.plot(np.linspace(0,eda_a.shape[0], eda_a.shape[0]), eda_a)
#plt.plot(np.linspace(0, temp_a.shape[0], temp_a.shape[0]), temp_a)
print("EDA SHAPE BEFORE INTERP", eda_a.shape)
eda_a = reshape_array_freq(eda_a[1],bvp_a[1],eda_a)
print("EDA SHAPE AFTER INTERP", eda_a.shape)
#plotting bvp
bvp_a = bvp_a[2:]
eda_a = eda_a[2:]
bvp_a, eda_a = resize_ary(bvp_a, eda_a)

plt.plot(np.linspace(0, eda_a.shape[0], eda_a.shape[0]), eda_a)
plt.plot(np.linspace(0,bvp_a.shape[0], bvp_a.shape[0]), bvp_a)
plt.show()

data = np.array((eda_a, temp_a))
data = data.T

print(data[0])
print(data.shape)

labels = [[0, 0], [30, 1], [60, 0], [90, 1]]
print(labelize(freq=4, labels=labels).shape)
