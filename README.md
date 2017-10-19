# Stress detect - Deep learning classification
## He-Arc - Third project
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
  * Data pre-processing / 
  
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
