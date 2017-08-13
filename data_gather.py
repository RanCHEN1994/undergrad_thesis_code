#import exp_data11 as dat11
from exp_data11 import *
# std mean convert
M_S=np.zeros([len(neural_net['C_E']),4])

for i in range(len(neural_net['C_E'])):
	M_S[i,0]=neural_net['C_E'][i].mean()
	M_S[i,1]=neural_net['C_E'][i].std()
	M_S[i,2]=regression_1['A'][i][-1][0]
	M_S[i,3]=regression_2['A'][i][-2][0]

import matplotlib.pyplot as plt
epoch=neural_net['epoch']
epoch[20]=60000
#plt.axis([0,30000,8,11])
plt.axis([0,60000,6,12])
nn,=plt.plot(epoch,M_S[:,0],'r*',label="Neural_Net")
#plt.plot(epoch,M_S[:,2],'g--',label="")
ar,=plt.plot(epoch,M_S[:,3],'b*',label="Adjust_Reg")
plt.title('Treatment Effect Estimation---Adjusted regression')
plt.legend([nn,ar],['Neural Net','Adjusted regression'])
plt.plot(epoch,10*np.ones([21]))
plt.xlabel('epoch', fontsize=14, color='red')
plt.ylabel('Treatment Effect Estimation', fontsize=14, color='red')
#plt.yticks([-1000,-500,11,500,1000,1500],['-1000','-500','11','500','1000','1500'])
plt.show()

for i in range(len(neural_net['cost'])):
	regression_2['loss'][i]=regression_2['loss'][i][0][0]
plt.axis([0,60000,0,1200])
ar,=plt.plot(epoch,regression_2['loss'],'b*',label="Adjust_Reg")
nn,=plt.plot(epoch,neural_net['cost'],'r*',label="Neural_Net")
plt.legend([nn,ar],['Neural Net','Adjusted regression'])
plt.xlabel('epoch', fontsize=14, color='red')
plt.title('Cost')
plt.ylabel('cost', fontsize=14, color='red')
plt.show()


plt.plot(epoch,M_S[:,1],'*')
plt.title('Estimator-Standard deviation-NN')
plt.xlabel('epoch',fontsize=14)
plt.ylabel('Treatment Effect Estimation',fontsize=14)
plt.show()

