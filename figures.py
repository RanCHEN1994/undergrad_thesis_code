import numpy as np
import exp_data4 as dat4
import exp_data4 as dat5
import matplotlib.pyplot as plt
'''
variable=[]
for i in range(len(dat4.neural_net['variable'])):
	variable.append(dat4.neural_net['variable'][i][0])
#variable=np.matrix(variable)
print (type(variable[0]))
'''

loss1=dat5.regression_1['loss']
loss2=dat5.regression_2['loss']

for i in range(len(loss1)):
	loss1[i]=loss1[i][0][0]
	loss2[i]=loss2[i][0][0]


