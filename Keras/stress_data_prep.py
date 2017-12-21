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
def reshape_array_freq(bFreq, freq, ary):
    if bFreq is freq:
        return ary
    else:
        dF = int(freq/bFreq)
        new = np.empty((ary.shape[0]-1) * int(dF))
        for i in range(len(ary)-1):
            delta = (ary[i+1] - ary[i])/dF
            for c in range(int(dF)):
                new[(i*dF)+c] = ary[i] + delta*c
        return new

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
#eda_b = np.delete(eda_b, np.s_[5000:8500], axis = 0)
#temp_b = np.delete(temp_b, np.s_[5000:8500], axis = 0)
#timestamps1 = [(0, 1500), (5310, 8150), (11425, 12493), (16170, 17360), (20800, 21813), (25316, -1)]

temp_freq = temp_a[1]
hr_freq = hr_a[1]
bvp_freq = bvp_a[1]
eda_freq = eda_a[1]

#stripping useless infos
bvp_a = bvp_a[2:]
eda_a = eda_a[2:]
hr_a = hr_a[2:]
temp_a = temp_a[2:]
#interpolating missing values
eda_a = reshape_array_freq(eda_freq,bvp_freq,eda_a)
hr_a = reshape_array_freq(hr_freq, bvp_freq, hr_a)
temp_a = reshape_array_freq(temp_freq, bvp_freq, temp_a)


bvp_a, eda_a = resize_ary(bvp_a, eda_a)
bvp_a, hr_a = resize_ary(bvp_a, hr_a)
bvp_a, temp_a = resize_ary(bvp_a, temp_a)

plt.plot(np.linspace(0, eda_a.shape[0], eda_a.shape[0]), eda_a)
plt.plot(np.linspace(0,bvp_a.shape[0], bvp_a.shape[0]), bvp_a)
plt.plot(np.linspace(0,hr_a.shape[0], hr_a.shape[0]), hr_a)
plt.plot(np.linspace(0,temp_a.shape[0], temp_a.shape[0]), temp_a)
plt.show()

data = np.array((eda_a, temp_a))
data = data.T
print(data)
