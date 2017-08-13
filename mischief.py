import numpy as np
import tensorflow as tf

a=np.array([[1,2,3],[4,5,6]])
shape=tf.shape(a)

with tf.Session() as sess:
	sess.run(shape)
	print('shape',type(shape))
	print('shape2',shape.eval()[0],shape.eval()[1])
