# Import the liberaries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
import re, glob, csv

# fatch all the file in one list
all_files = glob.glob("DS/*.csv")
# set epoch 
epoch = 50
# Iterate through all the files and get all input integer values in the list
x=[]
for filename in all_files:
    # read csv file through pandas and remove header
	df = pd.read_csv(filename,index_col=None, header=0)
    # convert it into list
	l = df.values.tolist()
	# replace special character in to space
	str1 = str(l).strip('[]')
	str1 = str1.replace("[",' ').replace("]",' ').replace("'",'').replace("   "," ").replace("  "," ").replace(",","")
    # get only integer value
	c = [int(s) for s in str1.split() if s.isdigit()]
    # append in list x
	x.append(c)
# convert list to array
d_arr = np.array(x)
	
# Iterate through all the files and get objective values in single list
c =[]
for filename in all_files:
	df = pd.read_csv(filename)
	k = df.columns
	for i in k:
        # find all the value that start with integer and store into list
		temp = re.findall(r'\d+\.\d+', i) 
	c.append(temp)
l = []
# conver the list values in to the integer and append in list
for i in c:
	for j in range(len(i)):
		c = float(i[j])
		l.append(int(c))
# convert list into integer
s_arr = np.array(l)
#print(s_arr.shape)
# split input and output data into train and test in 80 to 20 ratio.
X_train, X_test, y_train, y_test = train_test_split(d_arr, s_arr, test_size = 0.2)

# feature scaling 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# reshape the input values into 3-D.
x_train = X_train_scaled.reshape(800,2500,1)
x_test = X_test_scaled.reshape(200,2500,1)

################### ANN
#Initialize model
ann = Sequential()      
#input layer                                      
ann.add(Dense(128, activation = 'relu', input_dim = 2500))  
#hidden layer-1  
ann.add(Dense(64, activation = 'relu')) 
#hidden layer-2             
ann.add(Dense(32, activation = 'relu'))
 #output layer               
ann.add(Dense(1))                                    
# Compile the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')  
# Pass the input data into model
ann.fit(X_train_scaled, y_train, batch_size = 32,validation_data=(X_test_scaled, y_test), epochs = epoch, verbose = 1)
# Save the weights of ANN model
ann.save("1111092-ANN.h5")
# Evaluate performance score on test data
scores = ann.evaluate(X_test_scaled, y_test, verbose=0)

################ CNN
#Initialize model 
cnn = Sequential()  
#Convolution Layer-1                                                                      
cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(2500,1)))
#Convolution Layer-2
cnn.add(Conv1D(filters=16, kernel_size=3, activation='relu'))  
#Dropout Layer                         
cnn.add(Dropout(0.5))            
#Maxpooling Layer                                                     
cnn.add(MaxPooling1D(pool_size=2)) 
#Flatten Layer                                                    
cnn.add(Flatten())     
#Dense Layer                                                               
cnn.add(Dense(units = 1))  
#Compile the model                                                           
cnn.compile(optimizer = 'adam', loss = 'mean_squared_error')                          
#Fit data into model
cnn.fit(x_train, y_train, batch_size = 32,validation_data=(x_test, y_test), epochs = epoch, verbose = 1)
#Save the weights
cnn.save("1111092-CNN.h5")
#Evaluate the model performance
scores = cnn.evaluate(x_test, y_test, verbose=0)

####### ANN
#initialize Model
ann_new = Sequential()
#Input Layer                                                  
ann_new.add(Dense(128, activation = 'relu', input_dim = 2500))
#Hidden Layer-1
ann_new.add(Dense(64, activation = 'relu'))
#Hidden Layer-2
ann_new.add(Dense(32, activation = 'relu'))
#Output Layer
ann_new.add(Dense(1))
#compile the model
ann_new.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Load the weights from the saved model
ann_new.load_weights("ANN.h5")   
# predict the data                                       
y_pred = ann_new.predict(X_test_scaled)

####### CNN
#Reconstruct identical new mod0.el
cnn_new = Sequential() 
#Convolution Layer-1                                                   
cnn_new.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(2500,1)))
#Convolution Layer-2
cnn_new.add(Conv1D(filters=16, kernel_size=3, activation='relu'))    
#Dropout Layer
cnn_new.add(Dropout(0.5))
#Maxpooling Layer
cnn_new.add(MaxPooling1D(pool_size=2))
#Flatten Layer
cnn_new.add(Flatten())
cnn_new.add(Dense(units = 1))
cnn_new.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Load the weights
cnn_new.load_weights("CNN.h5")                                            
Y_pred = cnn_new.predict(x_test)

# plotting the graph for ANN-----------------------------------
plt.plot(y_test , color = 'red', label = 'True data')
plt.plot(y_pred , color = 'green', label = 'Predicted data')
plt.title('ANN-Prediction graph')
plt.legend()
plt.show()
# plotting the graph for CNN-----------------------------------
plt.plot(y_test , color = 'red', label = 'True data')
plt.plot(Y_pred, color = 'green', label = 'Predicted data')
plt.title('CNN-Prediction CNN')
plt.legend()
plt.show()

print('############### ANN ###########################')
print('mean_squared_error:',mean_squared_error(y_test, y_pred))
print('mean_absolute_error:', mean_absolute_error(y_test, y_pred))
print('############### CNN ###########################')
print('mean_squared_error:', mean_squared_error(y_test, Y_pred))
print('mean_absolute_error:', mean_absolute_error(y_test, Y_pred))
