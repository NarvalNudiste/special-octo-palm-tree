import cv2
import tensorflow as tf
import keras

a = tf.truncated_normal([16,128,128,3])


classes = ['dogs', 'cats']
num_classes = len(classes)
 
train_path='data/train'
 
# validation split
validation_size = 0.2
 
# batch size
batch_size = 16
 
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(a)))