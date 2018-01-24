#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.callbacks import TensorBoard
#from sklearn.model_selection import StratifiedKFold
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import gc

MAX_FREQ = 64
writing = False

path_a = "data/URIT_1581_A01212B_2017_10_31_0946/"
path_b = "data/URIT_1581-A01713-2017_10_31_0949/"

class Person:
    def __init__(self, start, stop, overall_health, energetic, overall_stress, stressed_past_24h, sleep_quality_past_24h,  sleep_quality_past_month, id, reliable):
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
        self.tags = None
        self.binary_output = None
        self.multiclass_ouput = None
        self.reliable = reliable

    def prepare_data_binary(self):
        length = self.bvp.shape[0]
        self.correct_time()
        self.binary_output = np.empty(self.bvp.shape[0])
        for i in range(length):
            if i < self.tags[0]:
                self.binary_output[i] = 0.0
            else:
                self.binary_output[i] = 1.0

    def prepare_data_multiclass(self):
                length = self.bvp.shape[0]
                self.correct_time()
                self.multiclass_output = np.empty(self.bvp.shape[0])
                for i in range(length):
                    if i < self.tags[0]:
                        self.multiclass_output[i] = 0
                    elif i < self.tags[1]:
                        self.multiclass_output[i] = 1
                    elif i < self.tags[2]:
                        self.multiclass_output[i] = 2
                    else:
                        self.multiclass_output[i] = 3

    def correct_time(self):
        for i in range(0, len(self.tags)):
            self.tags[i] = self.tags[i] - self.timestamps[0]

    def pprint(self):
        plt.plot(np.linspace(0, self.eda.shape[0], self.eda.shape[0]), self.eda, label="EDA")
        plt.plot(np.linspace(0, self.bvp.shape[0], self.bvp.shape[0]), self.bvp, label="BVP")
        plt.plot(np.linspace(0, self.hr.shape[0], self.hr.shape[0]), self.hr, label="HR")
        plt.plot(np.linspace(0, self.temp.shape[0], self.temp.shape[0]), self.temp, label="temperature")
        for i in range(0, len(self.tags)):
            plt.plot([self.tags[i], self.tags[i]], [1, 1000], color = 'red', linewidth = 2.5, linestyle = "--")
        plt.show()
    def pprint_eda(self, min=None, max=None):
        plt.plot(np.linspace(0, self.eda.shape[0], self.eda.shape[0]), self.eda)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=1, mode="expand", borderaxespad=0.)
        for i in range(0, len(self.tags)):
                plt.plot([self.tags[i], self.tags[i]], [np.amin(self.eda), np.amax(self.eda)], color = 'red', linewidth = 2.5, linestyle = "--", label="EDA")
        plt.show()
    def pprint_bvp(self):
        plt.plot(np.linspace(0, self.bvp.shape[0], self.bvp.shape[0]), self.bvp)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
        for i in range(0, len(self.tags)):
            plt.plot([self.tags[i], self.tags[i]], [np.amin(self.bvp), np.amax(self.bvp)], color = 'red', linewidth = 2.5, linestyle = "--", label="BVP")
        plt.show()
    def pprint_temp(self):
        plt.plot(np.linspace(0, self.temp.shape[0], self.temp.shape[0]), self.temp)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
        for i in range(0, len(self.tags)):
            plt.plot([self.tags[i], self.tags[i]], [np.amin(self.temp), np.amax(self.temp)], color = 'red', linewidth = 2.5, linestyle = "--", label="Temperature")
        plt.show()
    def pprint_hr(self):
        plt.plot(np.linspace(0, self.hr.shape[0], self.hr.shape[0]), self.hr)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
        for i in range(0, len(self.tags)):
            plt.plot([self.tags[i], self.tags[i]], [np.amin(self.hr), np.amax(self.hr)], color = 'red', linewidth = 2.5, linestyle = "--", label="HR")
        plt.show()

''' Fix missing timestamps '''
def fix_subjects():
    for s in subjects:
        if s.reliable is 0:
            if s.id is 2:
                print("2")
                s.tags[0] = 1850
                s.tags[1] = 17175
                print(s.tags)

            elif s.id is 5:
                last = s.tags[1]
                s.tags[1] = 16165
                s.tags = np.append([27685], s.tags)
                s.tags = np.append([last], s.tags)
                print("5")
                print(s.tags)

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

def concatenateTime(ary_a, ary_b, timestart_a, timestart_b):
    for i in range(ary_a.shape[0]):
        ary_a[i] = ary_a[i] - timestart_a
    for i in range(ary_b.shape[0]):
        ary_b[i] = ary_b[i] - timestart_b

    for i in range(ary_b.shape[0]):
        ary_b[i] = ary_b[i]+ary_a[-1]

    new_ary = np.concatenate((ary_a, ary_b), axis=0)
    for i in range(new_ary.shape[0]):
        new_ary[i] = new_ary[i]*64
    return new_ary

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

subjects = list()
print("loading json labels ..")
labels_data = json.load(open('data/labels.json'))
for persons in labels_data["persons"]:
    subjects.append(Person(persons["time_start"], persons["time_stop"], persons["overall_health"], persons["energetic"], persons["overall_stress"], persons["stressed_past_24h"], persons["sleep_quality_past_24h"], persons["sleep_quality_past_month"], persons["id"], persons["reliable"]))


data_ary = list()

print("loading data from csv files ..")
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

timestart_a = temp_a[0]
timestart_b = temp_b[0]

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

print("resampling arrays ..")
#interpolating missing values
eda_a = reshape_array_freq(eda_freq,MAX_FREQ,eda_a)
hr_a = reshape_array_freq(hr_freq, MAX_FREQ, hr_a)
temp_a = reshape_array_freq(temp_freq, MAX_FREQ, temp_a)

eda_b = reshape_array_freq(eda_freq,MAX_FREQ,eda_b)
hr_b = reshape_array_freq(hr_freq, MAX_FREQ, hr_b)
temp_b = reshape_array_freq(temp_freq, MAX_FREQ, temp_b)

print("resizing arrays .. ")
bvp_a, eda_a = resize_ary(bvp_a, eda_a)
bvp_a, hr_a = resize_ary(bvp_a, hr_a)
bvp_a, temp_a = resize_ary(bvp_a, temp_a)

bvp_b, eda_b = resize_ary(bvp_b, eda_b)
bvp_b, hr_b = resize_ary(bvp_b, hr_b)
bvp_b, temp_b = resize_ary(bvp_b, temp_b)

print("concatening arrays .. ")
bvp = np.concatenate((bvp_a, bvp_b), axis=0)
eda = np.concatenate((eda_a, eda_b), axis=0)
hr = np.concatenate((hr_a, hr_b), axis=0)
temp = np.concatenate((temp_a, temp_b), axis=0)
timestamps = concatenateTime(timestamps_a, timestamps_b, timestart_a, timestart_b)


'''
plt.plot(np.linspace(0, eda.shape[0], eda.shape[0]), eda)
plt.plot(np.linspace(0,bvp.shape[0], bvp.shape[0]), bvp)
plt.plot(np.linspace(0,hr.shape[0], hr.shape[0]), hr)
plt.plot(np.linspace(0,temp.shape[0], temp.shape[0]), temp)
plt.plot(np.linspace(0, temp_b.shape[0], temp_b.shape[0]), temp_b)
for i in range(0, len(timestamps)):
    plt.plot([timestamps[i], timestamps[i]], [22, 34], color = 'red', linewidth = 2.5, linestyle = "--")
plt.show()
'''

#workspace time
'''
plt.plot(np.linspace(0, temp_a.shape[0], temp_a.shape[0]), temp_a)
for i in range(0, len(timestamps_a)):
    timestamps_a[i] = int((timestamps_a[i] - timestart_a)*64)
    plt.plot([timestamps_a[i], timestamps_a[i]],[22, 34], color='red', linewidth=2.5, linestyle="--")
plt.show()'''
#plotting temp_a with timestamps
'''
plt.plot(np.linspace(0, temp_b.shape[0], temp_b.shape[0]), temp_b)
for i in range(0, len(timestamps_b)):
    timestamps_b[i] = int((timestamps_b[i] - timestart_b)*64)
    plt.plot([timestamps_b[i], timestamps_b[i]],[22, 34], color='red', linewidth=2.5, linestyle="--")
plt.show()'''
#end

if writing == True:
    #creating folders
    for i in range(0, len(subjects)):
        path = 'data/individual_data/{:d}'.format(i)
        if not os.path.exists(path):
            os.makedirs(path)
#stocking data in person class and writing data
#print(timestamps)
print("storing data in Person class")
for s in subjects:
    s.eda = eda[s.timestamps[0]:s.timestamps[1]]
    s.hr = hr[s.timestamps[0]:s.timestamps[1]]
    s.temp = temp[s.timestamps[0]:s.timestamps[1]]
    s.bvp = bvp[s.timestamps[0]:s.timestamps[1]]
    s.tags = timestamps[np.where(np.logical_and(timestamps>=s.timestamps[0],timestamps<=s.timestamps[1]))]
    print("generating binary classes")
    s.prepare_data_binary()
    #print("generating multiclass")
    #s.prepare_data_multiclass()
    #print("Subject id :", s.id, " -> tags = ", s.tags)
    if writing == True:
        np.savetxt('data/individual_data/{:d}/eda.csv'.format(s.id), s.eda, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/bvp.csv'.format(s.id), s.bvp, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/temp.csv'.format(s.id), s.temp, delimiter=' ')
        np.savetxt('data/individual_data/{:d}/hr.csv'.format(s.id), s.hr, delimiter=' ')


eda_a = None
eda_b = None
hr_b = None
hr_a = None
temp_a = None
temp_b = None
eda_a = None
eda_b = None
gc.collect()

#for s in subjects:
#    s.correct_time()

fix_subjects()
import labelizer
labelizer.labelize(subjects)




#for s in subjects:
    #if s.id is 2 or s.id is 5:
        #s.pprint_eda()
'''
for s in subjects:
    if s.id == 1:
        s.pprint_eda(0.0, 0.2)

for s in subjects:
    s.pprint_temp()

for s in subjects:
    s.pprint_bvp()
'''


'''

bvp, eda = resize_ary(bvp, eda)
bvp, hr = resize_ary(bvp, hr)
data = np.array((bvp, eda, hr))
data = data.T

print("data loaded.")
'''
