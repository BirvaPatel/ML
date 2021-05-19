import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
######################################load the dataset####################################################
df = pd.read_csv('sobar-72.csv', header=None)
#print(df.head)
#print(df.columns.values)
x = df.iloc[1:, 13:19].values
Y = df.iloc[1:, -1].values 
#print(Y.shape) #(72, )
#print(Y)
#print(x.shape) # (72,5)
#print(x)

S = np.zeros((72, 7))
S[:, :6] = x
S[:, 6] = Y
random.shuffle(S)
print(S.shape)
######################################Split the dataset###################################################

train = S[:50]
print(train.shape)
#print(train)
test = S[50:]
print(test.shape)
#print(test)

x_ttrain = train[:,0:6]
print(x_ttrain.shape)
#print(x_ttrain)
y_train = train[:,6]
print(y_train.shape)
#print(y_train)
x_ttest = test[:,0:6]
print(x_ttest.shape)
y_test = test[:,6]
print(y_test.shape) 
# Normalize the training data
norm = MinMaxScaler().fit(x_ttrain)
x_train = norm.transform(x_ttrain)
#print(x_train)
x_test = norm.transform(x_ttest)

######################################################training the data#################################
Th = 0.0     # threshold value
eta = 0.1  # learning rate value
num_iter = 50   # number of iterations
epoch = 50 #num of epoch

def perceptron_train(x, y, Th, eta, num_iter):

        # Initializing parameters for the Perceptron
        w = np.zeros(len(x[0]))        # initial weights
        a = 0                          
        y_vec = np.ones(len(y))     # predictions
        e = np.ones(len(y))       # initializing error vector
        SE = []                         # initializing for sum of squared error.

        while a < num_iter:                             
            for i in range(0, len(x)):                 
                f = np.dot(x[i], w)   #sum of (Xi.w)       
                if f >= Th:     #activation                          
                    yhat = 1.                               
                else:                                   
                    yhat = 0.
                y_vec[i] = yhat
                for j in range(0, len(w)):    # weight updation process          
                    w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]
            a += 1          
            for i in range(0,len(y)):     # Sum of squared errors
               e[i] = (y[i]-y_vec[i])**2
            SE.append(0.5*np.sum(e))
        return w, SE

perceptron_train(x_train, y_train, Th, eta, num_iter)

w = perceptron_train(x_train, y_train, Th, eta, num_iter)[0]
SE = perceptron_train(x_train, y_train, Th, eta, num_iter)[1]
print ("The sum-of-squared errors are:",SE)
################################################testing the data###########################################
def perceptron_test(x, w, Th, eta, num_iter):
	y_pred = []
	for i in range(0, len(x-1)):
		f = np.dot(x[i], w)   #sum of (Xi.w)
		if f > Th:               #activation                
			yhat = 1                               
		else:                                   
			yhat = 0
		y_pred.append(yhat)
	return y_pred

y_pred = perceptron_test(x_test, w, Th, eta, num_iter)
#print(y_pred)

print ("accuracy", accuracy_score(y_test, y_pred))

