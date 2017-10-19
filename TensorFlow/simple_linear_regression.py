import tensorflow as tf
import cv2
import numpy as np
 
hr_data = np.genfromtext('../SVM/data/HR.csv', delimiters = ' ')
eda_data = np.genfromtext('../SVM/data/EDA.csv', delimiters = ' ')
stress_data = np.genfromtext('../SVM/data/STRESS.csv', delimiters = ' ')

trainStressX = 

trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

print(trainX)
print(trainY)

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)

cost = (tf.pow(Y - y_model, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        for (x,y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X:x, Y:y})
    
    print(sess.run(w))
