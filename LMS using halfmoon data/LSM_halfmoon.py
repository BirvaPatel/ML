# ML home assignment-2 part-1.
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time
from math import sqrt
###################################################Generating the halfmoon data###############################
# number of data point for each half moon.
n_data_points = 3500
# initializing the value of radius thickness and distance.
R = 10
num_iter = 5
D = 1

L1 = np.array([(R+num_iter)/2, D/2]) # (2, )
L2 = np.array([-(R+num_iter)/2, -D/2]) # (2, )

R_1 = np.random.rand(n_data_points)*num_iter+R     #initiating the radius from [R, R+num_iter]
A_1 = np.random.rand(n_data_points)*np.pi   #Defining angels from 0 to pi.

R_2 = np.random.rand(n_data_points)*num_iter+R
A_2 = np.random.rand(n_data_points)*np.pi+np.pi

C_1 = np.array((R_1*np.cos(A_1), R_1*np.sin(A_1)))  # (2,3000) #for ploting the data, converting it in cartesian.
C_2 = np.array((R_2*np.cos(A_2), R_2*np.sin(A_2)))  # (2,3000)

x1= np.zeros((3500,2)) # gathering the points of the double moon.
x2= np.zeros((3500,2))
x1[:,0],x1[:,1] = (C_1[0] - L1[0], C_1[1] - L1[1])
x2[:,0],x2[:,1] = (C_2[0] - L2[0], C_2[1] - L2[1])

Y = np.zeros(7000) # defining numpy array to store labels.
Y[:3500] = 0
Y[3500:] = 1

S = np.zeros((7000, 3)) # combining both features and labels in array S.
S[:, :2] = np.concatenate((x1,x2))
S[:, 2] = Y
random.shuffle(S) #shuffling the data.
print(S.shape)
##############################################  plotting the graph  #####################################
X = S[:, :2]  # dividing the data to see plotted graph of halfmoon.
Y = S[:, 2]
# graph for half moon.
plt.scatter(X[:, 0], X[:, 1], c=Y, marker='*', linewidths=0.1)
plt.show()
##############################################  spliting the dataset  ###############################
# spliting the dataset to train and test.
train = S[:5000]
print(train.shape)
test = S[5000:]
print(test.shape)

# now furthur spliting for x_train, y_train, x_test, y_test
x_ttrain = train[:,0:2]
print(x_ttrain.shape)
y_train = train[:,2]
print(y_train.shape)
x_ttest = test[:,0:2]
print(x_ttest.shape)
y_test = test[:,2]
print(y_test.shape) 

# Normalize the training and testing data
norm = MinMaxScaler().fit(x_ttrain)
x_train = norm.transform(x_ttrain)
#print(x_train)
x_test = norm.transform(x_ttest)

############################################ traing the dataset ################################
Th = 0.0     # threshold value
eta = 0.1   # learning rate value
num_iter = 4     # number of iterations
epoch = 10 # epoch
correct_train = 0.0

def LMS_train(x, y, Th, eta, num_iter):

		# Initializing parameters for the Perceptron
		w = np.random.randn(len(x[0]))* 0.01		# initial weights
		a = 0                          
		y_predict = np.ones(len(y))     # predictions
		'''		
		# Make copies of the original data
		train_X_unshuffled = x.copy()
		train_y_unshuffled = y.copy()
		idx = np.arange(x.shape[0])
		'''
		for p in range(epoch):
			'''
			print(p)
			random.shuffle(idx)
			x = train_X_unshuffled[idx]
			y = train_y_unshuffled[idx]
			'''
			while a < num_iter:
				#print(a)
				for i in range(0, len(x)):                 
					f = np.dot(x[i], np.transpose(w))   #sum of (Xi.w)       
					if f >= Th:           #activation                              
						y_p = 1.                               
					else:                                   
						y_p = 0.
					y_predict[i] = y_p
					for j in range(0, len(w)):  # weight updation process           
						w[j] = w[j] + eta*(y[i]-y_p)*x[i][j]
				a += 1	
		return w
start = time.time()
w = LMS_train(x_train, y_train, Th, eta, num_iter)
stop = time.time()

################################################ testing the dataset ###################################
def LMS_test(x, w, Th, eta, num_iter):
	y_pred = []
	for i in range(0, len(x-1)):
		f = np.dot(x[i], np.transpose(w))    #sum of (Xi.w)
		if f > Th:           #activation                    
			y_p = 1                               
		else:                                   
			y_p = 0
		y_pred.append(y_p)
	return y_pred
T_start = time.time()
y_pred = LMS_test(x_test, w, Th, eta, num_iter)
tra_acc = LMS_test(x_train,w,Th,eta,num_iter)
T_stop = time.time()

################################################### Results ###########################

print('******************Training Results*******************')
print(f"Training time: {stop - start}s")
print ("training accuracy: ", accuracy_score(y_train, tra_acc))
train_rmse = sqrt(mean_squared_error(y_train, tra_acc))
print("training RMSE: ",train_rmse)

print("****************Testing Results*********************")

print(f"Testing time: {T_stop - T_start}s")
print ("testing accuracy: ", accuracy_score(y_test, y_pred))
test_rmse = sqrt(mean_squared_error(y_test, y_pred))
print("testing RMSE: ",test_rmse)


'''
RMSE = [] 
def RootMSE(y, y_predict):
	e = np.ones(len(y))       # initializing error vector
	for i in range(0,len(y)):     # Sum of squared errors
		e[i] = (y[i]-y_predict[i])**2
	RMSE.append(np.sqrt(np.sum(e)/float(len(y))))
	return RMSE
testing_rm = RootMSE(y_test, y_pred)
print("testing rmse", testing_rm)			
'''