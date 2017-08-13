#!/Users/chenran/anaconda/envs/tensorflow/bin/python

import numpy as np
import tensorflow as tf

n=10
x=tf.placeholder('float',[None,n])
y=tf.placeholder('float')
batch_size=100

neural_net={'epoch':[],'C_E':[],'real':[],'pred':[],'variable':[],'cost':[],'variable_name':[]}
regression_1={'epoch':[],'err':[],'A':[],'pred':[],'loss':[]}
regression_2={'epoch':[],'err':[],'A':[],'pred':[],'loss':[]}

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

def data_generate_polynomial(m):		#m data points, 9 attributes(0,...,8)
	a=np.zeros([m,11])
	parameter=np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]) #first line is for linear,second for quadratic
	for i in range(m):							
		a[i:(i+1),:5]=np.random.normal(0,1,5)			#0,1,2,3,4 attributes obey gaussian
		a[i:(i+1),5:9]=np.random.uniform(5,10,4)		#5,6,7,8 attributes obey uniform
		a[i:(i+1),9:10]=np.random.randint(0,2)
		eps=float(np.random.normal(0,1,1))				#eps obey normal
		a[i,10]=np.dot(parameter[0:1,:],a[i,:10])+np.dot(parameter[1:,:],a[i,:10]*a[i,:10])+eps
	return a[:,:-1],a[:,-1:]

#layer_1={'weight':tf.Variable(tf.random_normal([n,1]),name='simple_w_1'),'bias':tf.Variable(0.0,name='simple_b_1')}
#def neural_network_model(x):		#linear model
#	l1=tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias'])
#	return l1
#


def neural_network_model_sig(x):		#linear model with sig
	layer_1={'weight':tf.Variable(tf.random_normal([n,1]),name=sig_w_1),'bias':tf.Variable(0.0,name=sig_b_1)}
	layer_2={'ratio':tf.Variable(1.0,name=sig_r_2),'bias':tf.Variable(0.0,name=sig_b_2)}
	layer_3={'ratio':tf.Variable(500.0,name=sig_r_3),'bias':tf.Variable(0.0,name=sig_b_3)}
#l1=tf.mul(0.01,tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias']))
	#l1=tf.nn.sigmoid(l1)
	#layer_2={'ratio':tf.Variable(200.0),'bias':tf.Variable(0.0)}
	#l2=tf.mul(tf.add(l1,layer_2['bias']),layer_2['ratio'])
	l1=tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias'])
	l_nmlz=l1-tf.reduce_mean(l1)
	l_std=tf.nn.l2_normalize(l_nmlz,dim=0)
	l2=tf.mul(layer_2['ratio'],l_std)+layer_2['bias']
	l3=tf.nn.sigmoid(l2)
	l4=tf.mul(layer_3['ratio'],l3)+layer_3['bias']
	return l4,l3



#layer_1={'weight':tf.Variable(start1,name=sig2_w_1),'bias':tf.Variable(start2,name=sig2_b_1)}
#layer_2={'ratio':tf.Variable(200.0,name=sig2_r_2),'bias':tf.Variable(0.0,name=sig2_b_2)}
#def neural_network_model_sig2(x,start1,start2):		#linear model with sig
#	l1=tf.mul(tf.Variable((np.dot(np.dot(x,start1)+start2,np.dot(x,start1)+start2))**(-1),name=sig2_r_1),tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias']))
#	l1=tf.nn.sigmoid(l1)
#	l2=tf.mul(tf.add(l1,layer_2['bias']),layer_2['ratio'])
#	return l2,l1


#layer_1={'ratio':tf.Variable(np.asarray([[2.0],[4.0],[6.0],[8.0],[10.0],[24.0],[28.0],[32.0],[36.0],[10.0]]),name='sig3_r_1',dtype='float32'),'bias':tf.Variable(15.0,name='sig3_b_1')}
##layer_1={'ratio':tf.Variable(tf.convert_to_tensor(np.array([2.0,4.0,6.0,8.0,10.0,6.0,7.0,8.0,9.0,5.0]),dtype=float32),name='sig3_r_1'),'bias':tf.Variable(0.0,name='sig3_b_1')}
#layer_2={'ratio':tf.Variable(0.5)}
#def neural_network_model_sig3(x):
#	l0=x-tf.reduce_mean(x,0)
#	l0=tf.nn.l2_normalize(l0,0)
#	l0=tf.mul(layer_2['ratio'],l0)
#	l0=tf.nn.sigmoid(l0)
#	l1=tf.add(tf.matmul(l0,layer_1['ratio']),layer_1['bias'])
#	return l1
	
def startpoint(x,y):
	err,A,pred,loss = regression(x,y)
	return A[:(-1)],A[(-1):]


layer_1={'weight':tf.Variable(tf.random_normal([n,4]),name='model2_w_1'),'bias':tf.Variable(tf.random_normal([1,4]),name='model2_b_1')}
layer_2={'weight':tf.Variable(tf.random_normal([4,1]),name='model2_w_2'),'bias':tf.Variable(0.0,name='model2_b_2')}
layer_3={'ratio':tf.Variable(1.0,name='layer3')}
def neural_network_model2(x):		#linear model with sigmoid
	l1=tf.add(tf.matmul(x,layer_1['weight']),layer_1['bias'])
	l1=l1-tf.reduce_mean(l1,0)
	l1=tf.nn.l2_normalize(l1,0)
	l1=tf.mul(layer_3['ratio'],l1)
	l2=tf.nn.sigmoid(l1)
	l3=tf.add(tf.matmul(l2,layer_2['weight']),layer_2['bias'])
	return l3

def causal_effect_pre(x):  
#	attribute,indicator=tf.split(tf.convert_to_tensor(x),[int(9),int(1)],1)
	dim=tf.shape(x)
	attribute=tf.slice(x,[0,0],[dim[0],(dim[1]-1)])
	treat=tf.ones(shape=(dim[0],1))
	control=tf.zeros(shape=(dim[0],1))
	T_group=tf.concat(1,[attribute,treat])
	C_group=tf.concat(1,[attribute,control])
	print(type(x))
#	attribute=x[:,:9]
#	indicator=x[:,9:10]
#	treat=np.ones([tf.shape(attribute)[0],1])
#	control=np.zeros([tf.shape(attribute)[0],1])
#	T_group=np.concatenate((attribute,treat),axis=1)
#	C_group=np.concatenate((attribute,control),axis=1)
	return T_group,C_group

def C_E_nn(x):
	T_group,C_group=causal_effect_pre(x)	
#	effect=neural_network_model(tf.cast(T_group,tf.float32))-neural_network_model(tf.cast(C_group,tf.float32))
	T=tf.concat(0,[T_group,C_group])
	effect_raw=neural_network_model2(T)
	dim=tf.shape(T_group)[0]
	effect=tf.slice(effect_raw,[0,0],[dim,1])-tf.slice(effect_raw,[dim,0],[dim,1])
	#tf.Print(T_group,[T_group,'T'])
	#tf.Print(C_group,[C_group,'C'])
	#tf.Print(effect,[effect,'E'])
	return effect,T_group,C_group

def train_neural_network(x):
	prediction = neural_network_model2(x)
	#prediction,sig = neural_network_model_sig(x)
	#prediction=neural_network_model_sig3(x)
#	prediction=neural_network_model_sig2(x,startpoint(x,y))

	#cost=tf.reduce_mean(tf.square(tf.sub(x,y)))
	cost = tf.reduce_mean(tf.squared_difference(prediction,y))
	optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)  #usable
#	optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(cost)	
	csef,T,C=C_E_nn(x)
	check1=neural_network_model2(T)
	check2=neural_network_model2(C)
	
	hm_epoch = 60000

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		print('into tensorflow')	
		for epoch in range(hm_epoch):
			epoch_loss = 0
			epoch_x,epoch_y = data_generate_linear(200) 
		#	epoch_x,epoch_y = data_generate_polynomial(200) 
			#_,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
			_,c,csef1,prediction1,T1,C1,check_1,check_2 = sess.run([optimizer,cost,csef,prediction,T,C,check1,check2],feed_dict={x:epoch_x,y:epoch_y})
			#_,c,sig1 = sess.run([optimizer,cost,sig],feed_dict={x:epoch_x,y:epoch_y})
		#	print('prediction',prediction1.shape)
		#	print('inter',inter1.shape)
		#	for i in tf.trainable_variables():
		#		print(i.name)
			epoch_loss = c
			if epoch%3000==0:
				print('Epoch',epoch,'completed out of',hm_epoch,'loss:',epoch_loss)
#				print('sigmoid_function',sig1)
	#			print('T',T1.shape)
	#			print('C',C1)
	#			print('E',csef1)
	#			print('check1',check_1)
	#			print('check2',check_2)
				neural_net['epoch'].append(epoch)
				regression_1['epoch'].append(epoch)
				regression_2['epoch'].append(epoch)
				neural_net['C_E'].append(csef1)
				neural_net['cost'].append(epoch_loss)
				v=tf.trainable_variables()
				new=[]
				for i in v:
					new.append(i.eval())
#				for i in v:
#					print(i.__dict__)
#					print('\n')	
				neural_net['variable'].append(new)
				neural_net['pred'].append(prediction1)
				neural_net['real'].append(epoch_y)
				err,A,pred,loss=regression(epoch_x,epoch_y)
				regression_1['err'].append(err)
				regression_1['A'].append(A)
				regression_1['pred'].append(pred)
				regression_1['loss'].append(loss)
				err,A,pred,loss=regression2(epoch_x,epoch_y)
				regression_2['err'].append(err)
				regression_2['A'].append(A)
				regression_2['pred'].append(pred)
				regression_2['loss'].append(loss)

#neural_net={'epoch':[],'C_E_tr':[],'C_E_te':[],'pred':[],'variable':[],'lost':[]}
# 12 regression1={'epoch':[],'err':[],'A':[],'pred':[],'loss':[]}
# 13 regression2={'epoch':[],'err':[],'A':[],'pred':[],'loss':[]}
		test,result=data_generate_linear(200)
		#test,result=data_generate_polynomial(200)
		print('real_result:',result)
		print('prediction:',prediction.eval({x:test,y:result}))
		print('cost:',cost.eval({x:test,y:result}))
#		layer_1=tf.get_variable('layer_1',trainable=True)
#		print('weight:',layer_1['weight'],'bias:',layer_1['bias'])
#		tf.print('weight:',layer_1['weight'],'bias:',layer_1['bias'])
		err,A,pred,loss=regression(test,result)
		print('regression:','\n','pred:',pred,'err:',err,'loss:',loss,'A:',A)
		v=tf.trainable_variables()
		print('len(v)',len(v))
		print('v0,v1 and v2 are as follows:')
		v0=v[0]
		v1=v[1]
	#	v2=v[2]
		print v0.eval()
		print v1.eval()
	#	print v2.eval()
		epoch=-1
		neural_net['epoch'].append(epoch)
		regression_1['epoch'].append(epoch)
		regression_2['epoch'].append(epoch)
		neural_net['C_E'].append(csef.eval({x:test,y:result}))
		neural_net['cost'].append(cost.eval({x:test,y:result}))
		new=[]
		for i in v:
			new.append(i.eval())
			neural_net['variable_name'].append(i.name)
		neural_net['pred'].append(prediction.eval({x:test,y:result}))
		neural_net['real'].append(result)
		err,A,pred,loss=regression(test,result)
		regression_1['err'].append(err)
		regression_1['A'].append(A)
		regression_1['pred'].append(pred)
		regression_1['loss'].append(loss)
		err,A,pred,loss=regression2(test,result)
		regression_2['err'].append(err)
		regression_2['A'].append(A)
		regression_2['pred'].append(pred)
		regression_2['loss'].append(loss)
	
	
		with open ('./exp_data15.py','a+') as f:
			f.write('neural_net='+str(neural_net)+'\n')
			f.write('regression_1='+str(regression_1)+'\n')
			f.write('regression_2='+str(regression_2)+'\n')
	
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

def regression2(x,y):	#with intercept
	x=np.concatenate((x,np.ones([len(x),1])),axis=1)
	return regression(x,y)
	

train_neural_network(x)


