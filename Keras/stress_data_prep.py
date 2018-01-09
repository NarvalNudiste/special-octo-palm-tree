#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.callbacks import TensorBoard
#from sklearn.model_selection import StratifiedKFold
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from labelizer import labelize
import time
import json
from pprint import pprint

MAX_FREQ = 64
writing = False

path_a = "data/URIT_1581_A01212B_2017_10_31_0946/"
path_b = "data/URIT_1581-A01713-2017_10_31_0949/"
#TODO Séprarer en activités aussi, plus tard

class Person:
    def __init__(self, start, stop, overall_health, energetic, overall_stress, stressed_past_24h, sleep_quality_past_24h,  sleep_quality_past_month, id):
        self.timestamps = (start, stop)
        self.overall_health = overall_health
        self.energetic = energetic
        self.overall_stress = overall_stress
        self.stressed_past_24h = stressed_past_24h
        self.sleep_quality_past_24h = sleep_quality_past_24h
        self.sleep_quality_past_month = sleep_quality_past_month
        self.id = id
        self.eda = None
        self.hr = None
        self.temp = None
        self.bvp = None

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

subjects = list()
labels_data = json.load(open('data/labels.json'))
for persons in labels_data["persons"]:
    subjects.append(Person(persons["time_start"], persons["time_stop"], persons["overall_health"], persons["energetic"], persons["overall_stress"], persons["stressed_past_24h"], persons["sleep_quality_past_24h"], persons["sleep_quality_past_month"], persons["id"]))

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

timestamps_a = np.genfromtxt(path_a + "tags.csv", delimiter=",")
timestamps_b = np.genfromtxt(path_b + "tags.csv", delimiter=",")

timestart = temp_a[0]

for x in np.nditer(timestamps_b):
    #print(time.strftime('%H:%M:%S', time.localtime(x)))
    x = (x - temp_b[0])*4
    print(x)


#eda_b = np.delete(eda_b, np.s_[5000:8500], axis = 0)
#temp_b = np.delete(temp_b, np.s_[5000:8500], axis = 0)

temp_freq = temp_a[1]
hr_freq = hr_a[1]
bvp_freq = bvp_a[1]
eda_freq = eda_a[1]

#stripping useless infos
bvp_a = bvp_a[2:]
eda_a = eda_a[2:]
hr_a = hr_a[2:]
temp_a = temp_a[2:]
#b
bvp_b = bvp_b[2:]
eda_b = eda_b[2:]
hr_b = hr_b[2:]
temp_b = temp_b[2:]


#workspace time

plt.plot(np.linspace(0, temp_a.shape[0], temp_a.shape[0]), temp_a)
for i in range(0, len(timestamps_a)):
    timestamps_a[i] = int((timestamps_a[i] - timestart)*4)
    '''plt.annotate('time',
             xy=(timestamps_a[i], temp_a[int(timestamps_a[i]-2)]), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))'''
    plt.plot([timestamps_a[i], timestamps_a[i]],[22, 36], color='red', linewidth=2.5, linestyle="--")
plt.show()
#end




#interpolating missing values
eda_a = reshape_array_freq(eda_freq,MAX_FREQ,eda_a)
hr_a = reshape_array_freq(hr_freq, MAX_FREQ, hr_a)
temp_a = reshape_array_freq(temp_freq, MAX_FREQ, temp_a)

eda_b = reshape_array_freq(eda_freq,MAX_FREQ,eda_b)
hr_b = reshape_array_freq(hr_freq, MAX_FREQ, hr_b)
temp_b = reshape_array_freq(temp_freq, MAX_FREQ, temp_b)


bvp_a, eda_a = resize_ary(bvp_a, eda_a)
bvp_a, hr_a = resize_ary(bvp_a, hr_a)
bvp_a, temp_a = resize_ary(bvp_a, temp_a)

bvp_b, eda_b = resize_ary(bvp_b, eda_b)
bvp_b, hr_b = resize_ary(bvp_b, hr_b)
bvp_b, temp_b = resize_ary(bvp_b, temp_b)

bvp = np.concatenate((bvp_a, bvp_b), axis=0)
eda = np.concatenate((eda_a, eda_b), axis=0)
hr = np.concatenate((hr_a, hr_b), axis=0)
temp = np.concatenate((temp_a, temp_b), axis=0)


s_ary = np.empty((bvp.shape[0]))
for i in range(len(s_ary)):
    s_ary[i] = 1000 if i % (64*60) == 0 else 0

#temp_a, hr_a = resize_ary(temp_a, hr_a)plt.plot(np.linspace(0, s_ary.shape[0], s_ary.shape[0]), s_ary)

#plt.plot(np.linspace(0, eda.shape[0], eda.shape[0]), eda)
#plt.plot(np.linspace(0,bvp.shape[0], bvp.shape[0]), bvp)
#plt.plot(np.linspace(0,hr.shape[0], hr.shape[0]), hr)
#plt.plot(np.linspace(0,temp.shape[0], temp.shape[0]), temp)
#plt.plot(np.linspace(0, temp_b.shape[0], temp_b.shape[0]), temp_b)
#plt.show()

if writing == True:
    #creating folders
    for i in range(0, len(subjects)):
        path = 'data/individual_data/{:d}'.format(i)
        if not os.path.exists(path):
            os.makedirs(path)
#stocking data in person class and writing data
for s in subjects:
    s.eda = eda[s.timestamps[0]:s.timestamps[1]]
    s.hr = hr[s.timestamps[0]:s.timestamps[1]]
    s.temp = temp[s.timestamps[0]:s.timestamps[1]]
    s.bvp = bvp[s.timestamps[0]:s.timestamps[1]]
    if writing == True:
        np.savetxt('data/individual_data/{:d}/eda.csv'.format(s.id), s.eda, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/bvp.csv'.format(s.id), s.bvp, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/temp.csv'.format(s.id), s.temp, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/hr.csv'.format(s.id), s.hr, delimiter=' ')

data = np.array((bvp, eda, hr, temp))
data = data.T

print(data.shape)
