# Stress detection / Deep learning classification
## he-arc - 3rd project
##### Guillaume Noguera, inf3-dlma

### Stress classification with various machine learning libraries

This school project aims to explore machine learning algorithms through the use of SVM (scikit) and deep learning (keras, tensorflow).
Physiological data samples will be provided by the E4 *empatica* wristband.

### Introduction

Last year ago, a SVM model has been developed to classify stress levels according to physiological data.
The main aim of the project is to develop a model to see how well the deep learning approach can do in comparison of the svm approach.
In addition of this, the old dataset we have to work with hasn't been properly labelized - a new data collection is thus part of the project.
The old model will obviously use the same dataset to get relevant performance comparisons.

### Tensorflow / Keras

Keras now supports and uses Tensorflow (in addition of Theano) - it can be seen as an higher level library using tf - and will shortly be integrated to it. 
It can be used to quickly create complex models with minimal code. Tensorflow is more of a language than a framework, providing its own syntax to develop machine learning models.
While Tensorflow offers a greater degree of freedom, Keras is simpler and more user oriented. Like Scikit-learn, it provides pre-defined models (allowing users to define their own). 
A possible approach could be to use those models before diving into Tensorflow (as time could - and *will* - be a possible limitation).
Therefore, our main focus will be on Keras. 

### Requirements

1. SVM approach
  * Familiarization with Support Vector Machines (SVM)
  * Various tests with sci-kit's SVM classifier on provided sample data (iris, digits)
  * First implementation with small E4 datasets
  * Proper implementation with the actual database
	
2. Deep Learning approach
  * Familiarization with Deep Learning key concepts
  * Familiarization with Tensorflow and Keras libraries
  * Keras / Tensorflow comparison
  * Discussion of the final choice between Keras and Tensorflow

3. Getting data
  * Stress workshop planning
  * Actual data collection
  * Data pre-processing
  
4. Keras implementation
  * Model creation
  * Training and adjusments
  * (Optional) Tensorflow approach

5. Accuracy comparison 
  * Figuring a way to compare algorithms performance (False negative, false positive, etc.) 
  * Some visual representations
  * Preparing data for visualization
  * Coordination with the team
  
6. Documentation
  * Sphinx documentation
  * Ad-hoc LaTeX report
  
### Neural networks basics
  
At the core of every neural network is the perceptron, which dates back to the late 1950's. Invented by Frank Rosenblatt, the perceptron was largely inspired by neurobiology as it mimics neurons basic behaviour: a neuron takes an input and then choose to fire or not fire depending on input's value.
The function used to determine if a neuron is activated is called the activation function : it is often a non-linear function (Sigmoid, ArcTan, ReLU), as most real-world problems are non-linear indeed.

Perceptrons can produce one or several ouputs; they can can also be stacked, resulting in a multi-layer perceptron (MLP). 
The most basic MLP contains an input layer, an hidden layer and an output layer. As additionnals hidden layers are stacked on the top of each others, our basic MLP transitions into a deep neural network.

### Keras basics

Keras provides us with easy ways to quickly build a model : 

```python
model = Sequential()
```

Layers can then be stacked on top of each other this way : 

```python
model.add(Dense(32, input_shape=(*, 16))) # input arrays of shape (*, 16) and output arrays of shape (*, 32)
model.add(Dense(10, activation='softmax')) # activation function can be specified there
#and so on
```

Next, the model needs to be compiled. The optimizer, loss function and metrics are provided there.

```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
A lot of optimizers are available in Keras, such as stochastic gradient descent, RMSprop (often good for recurrent neural networks), ADAM .. the whole list is available in the [keras documentation.](https://keras.io/optimizers/) 

After compilation, the model can be trained & evaluated: 

```python
model.fit(data, labels, epochs=10, batch_size=32) #epochs are the number of passes
score = model.evaluate(x_test, y_test, batch_size=128)
```

### Performances Comparison

[Here's some performances comparison](docs/PERFS.md)

Thoses performances comparison comes from multiple training sessions on the pima diabetes dataset.
Models have been tested with different hidden layers and neurons numbers (from 1x1 to 64x64), with or without Dropout layers, and each time for 5 different optimizers.
Overall, the Rectifier activation function (ReLU) seems to be the best choice for binary classification on this dataset.
As this may vary on differnt data structure, I think repeating those tests with the proper data can help finding a suitable model. 


### Data collection / Workshop


### Data pre-processing

All E4 wristband sensors deliver different data formats, as the sample rate may vary : the photoplethysmograph sensor (BVP) has a 64Hz sample rate, whereas the electrodermal activity sensor sensor (EDA) only samples at 4Hz. 
Thus, we need a function using linear interpolation to even out our arrays  : 

```python
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
```

We also need to remove some data at the end of certain data arrays, as they do not have the same length after interpolation (I guess all the sensors don't necessarily stop at the exact same time)

```python

def resize_ary(a1, a2):
    diff = abs(a1.shape[0] - a2.shape[0])
    if a1.shape[0] < a2.shape[0]:
        a2 = a2[:-diff]
    else:
        a1 = a1[:-diff]
    return a1, a2
``` 

![Formatted data](Keras/pics/r_plot_all.png "Formatted data")


Our data being properly formatted, we then use JSON to store labels from the data collection survey.

```json
{
  "person" : {
    "id" : 0,
    "file" : "a",
    "time_start" :,
    "time_stop" :,
    "overall_health" : 2,
    "energetic" : 2,
    "overall_stress" : 3,
    "stressed_past_24h" : 3,
    "sleep_quality_past_24h" : 2,
    "sleep_quality_past_month" : 1
  }
}
```

Later on, those labels will hopefully help our classifier.

