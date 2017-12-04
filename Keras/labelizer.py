import numpy as np
#todo : gen from txt
def labelize(freq, labels):
    array = np.empty(0)
    currentLabel = -1
    for i in range(labels[-1][0]):
        for elem in labels:
            if i is elem[0]:
                currentLabel = elem[1]
        for k in range(0, freq):
            array = np.append(array, [currentLabel])
    return array
