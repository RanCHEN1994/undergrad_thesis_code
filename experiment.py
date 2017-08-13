#!/Users/chenran/anaconda/envs/tensorflow/bin/python

import numpy as np
import tensorflow as tf

n=10
x=tf.placeholder('float',[None,n])
y=tf.placeholder('float')
batch_size=100

def data_generate_linear(m):
	a=np.zeros([m,11])							#m items, 0-8 attributes,9 indicator, 10 response
	parameter=np.array([1,2,3,4,5,6,7,8,9,10])	#parameter of linear model
	for i in range(m):							
		a[i:(i+1),:5]=np.random.normal(0,10,5)			#0,1,2,3,4 attributes obey gaussian
		a[i:(i+1),5:9]=np.random.uniform(5,10,4)		#5,6,7,8 attributes obey uniform
		a[i:(i+1),9:10]=np.random.randint(0,2)
		eps=float(np.random.normal(0,1,1))				#eps obey normal
		a[i,10]=np.dot(parameter,a[i,:10])+eps
	return a[:,:-1],a[:,-1:]

def data_generate_polynomial(m,n):		#m data points, n attributes
	a=np.zeros([m,(n+2)])
	parameter=np.zeros([])
	for i in range(m):	
		pass

def neural_network_model(x):		#linear model
	layer_1={'weight':tf.Variable(tf.random_normal([n,1])),'bias':tf.Variable(0.0)}
	l1=tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias'])
	return l1

def neural_network_model2(x):		#linear model with relu
	pass

def train_neural_network(x):
	prediction = neural_network_model(x)
	#cost=tf.reduce_mean(tf.square(tf.sub(x,y)))
	cost = tf.reduce_mean(tf.squared_difference(prediction,y))
	optimizer=tf.train.AdamOptimizer(0.1).minimize(cost)
	
	print('line35')

	hm_epoch = 6000

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		print('into tensorflow')	
		for epoch in range(hm_epoch):
			epoch_loss = 0
			epoch_x,epoch_y = data_generate_linear(200) 
			_,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
			epoch_loss = c
			print('Epoch',epoch,'completed out of',hm_epoch,'loss:',epoch_loss)
		
		test,result=data_generate_linear(20)
		print('real_result:',result)
		print('prediction:',prediction.eval({x:test,y:result}))
		print('cost:',cost.eval({x:test,y:result}))
#		layer_1=tf.get_variable('layer_1',trainable=True)
#		print('weight:',layer_1['weight'],'bias:',layer_1['bias'])
#		tf.print('weight:',layer_1['weight'],'bias:',layer_1['bias'])
		err,A,pred,loss=regression(test,result)
		print('regression:','\n','pred:',pred,'err:',err,'loss:',loss,'A:',A)
		v=tf.trainable_variables()
		print('v0 and v1 are as follows:')
		v0=v[0]
		v1=v[1]
		print v0.eval()
		print v1.eval()
	
'''
bad code
		with tf.Session() as sess:
		v=tf.trainable_variables()
		w=sess.run(v)
		print('w0',w[0])
		print('w1',w[1])
'''		
def regression(x,y):
	H=np.mat(np.dot(x.T,x))
	A=np.dot(np.dot(H.I,x.T),y)
	pred=np.dot(x,A)
	err=y-pred
	loss=np.dot(err.T,err)/len(err)
	return err,A,pred,loss

train_neural_network(x)

#print ('Yes')
