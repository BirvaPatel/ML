import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
######################################### Generating the halfmoon data ###############################
# number of data point for each half moon.
n_data_points = 1500
# initializing the value of radius thickness and distance.
R = 9
num_iter = 5
D = 1

L1 = np.array([(R+num_iter)/2, D/2]) # (2, )
L2 =  np.array([-(R+num_iter)/2, -D/2]) # (2, )

R_1 = np.random.rand(n_data_points)*num_iter+R     #initiating the radius from [R, R+num_iter]
A_1 = np.random.rand(n_data_points)*np.pi   #Defining angels from 0 to pi.

R_2 = np.random.rand(n_data_points)*num_iter+R
A_2 = np.random.rand(n_data_points)*np.pi+np.pi

C_1 = np.array((R_1*np.cos(A_1), R_1*np.sin(A_1)))  # (2,3000) #for ploting the data, converting it in cartesian.
C_2 = np.array((R_2*np.cos(A_2), R_2*np.sin(A_2)))  # (2,3000)

x1= np.zeros((1500,2)) # gathering the points of the double moon.
x2= np.zeros((1500,2))
x1[:,0],x1[:,1] = (C_1[0] - L1[0], C_1[1] - L1[1])
x2[:,0],x2[:,1] = (C_2[0] - L2[0], C_2[1] - L2[1])

Y = np.zeros(3000) # defining numpy array to store labels.
Y[:1500] = 0
Y[1500:] = 1

S = np.zeros((3000, 3)) # combining both features and labels in array S.
S[:, :2] = np.concatenate((x1,x2))
S[:, 2] = Y
random.shuffle(S) #shuffling the data.
print(S.shape)
##############################################plotting the graph#####################################
X = S[:, :2]  # deviding the data to see plotted graph of halfmoon.
Y = S[:, 2]
# graph for half moon.
plt.scatter(X[:, 0], X[:, 1], c=Y, marker='*', linewidths=0.1)
plt.show()
############################################## spliting the dataset ###############################
# spliting the dataset to train and test.
train = S[:1000]
print(train.shape)
test = S[1000:]
print(test.shape)

# now furthur spliting for x_train, y_train, x_test, y_test
x_train = train[:,0:2]
print(x_train.shape)
y_train = train[:,2]
print(y_train.shape)
x_test = test[:,0:2]
print(x_test.shape)
y_test = test[:,2]
print(y_test.shape) 

############################################ traing the dataset ################################
Th = 0.0     # threshold value
eta = 0.01   # learning rate value
num_iter = 4     # number of iterations
epoch = 10 # epoch
def perceptron_train(x, y, Th, eta, num_iter):

        # Initializing parameters for the Perceptron
        w = np.zeros(len(x[0]))        # initial weights
        a = 0                          
        y_vec = np.ones(len(y))     # predictions
        e = np.ones(len(y))       # initializing error vector
        SE = []                         # sum of squared errors 

        while a < num_iter:                             
            for i in range(0, len(x)):                 
                f = np.dot(x[i], w)   #sum of (Xi.w)       
                if f >= Th:           #activation                              
                    yhat = 1.                               
                else:                                   
                    yhat = 0.
                y_vec[i] = yhat
                for j in range(0, len(w)):  # weight updation process           
                    w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]
            a += 1
            for i in range(0,len(y)):     # Sum of squared errors
               e[i] = (y[i]-y_vec[i])**2
            SE.append(0.5*np.sum(e))
        return w, SE
perceptron_train(x_train, y_train, Th, eta, num_iter)

w = perceptron_train(x_train, y_train, Th, eta, num_iter)[0]
SE = perceptron_train(x_train, y_train, Th, eta, num_iter)[1]
print ("The sum-of-squared e are:",SE)

################################################ testing the dataset ###################################
def perceptron_test(x, w, Th, eta, num_iter):
	y_pred = []
	for i in range(0, len(x-1)):
		f = np.dot(x[i], w)    #sum of (Xi.w)
		if f > Th:           #activation                    
			yhat = 1                               
		else:                                   
			yhat = 0
		y_pred.append(yhat)
	return y_pred
y_pred = perceptron_test(x_test, w, Th, eta, num_iter)
num_error = 0                   # misclassification count
for i in range(len(y_pred)):
	if y_pred[i] == y_test[i]:
		pass
	else:
		num_error += 1
################################################### Model Evaluation ###########################
print("number of mismatch points are: ", num_error)
print ("accuracy", accuracy_score(y_test, y_pred))

