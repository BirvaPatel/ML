# This is the code to calculate the Classification Accuracy.
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from elm import ELM
 

################################################# Loading dataset ###############################################

(train_data, train_labels),(test_data, test_labels) = cifar10.load_data()
train_data = train_data / 255.0
test_data = test_data / 255.0

train_X = train_data.astype('float32')
test_X = test_data.astype('float32')
Train_X = train_X.reshape(50000,3072)
Test_X = test_X.reshape(10000,3072)

train_Y = to_categorical(train_labels)
test_Y = to_categorical(test_labels)
print(train_Y.shape)
print(test_Y.shape)
print(Train_X.shape)
print(Test_X.shape)

#################################################### Model Implementation ##########################################
elm = ELM(200)
elm.fit(Train_X, train_Y)
# training predictions
start = time.time()	
tra_acc = elm.predict(Train_X)
stop = time.time()
	
#testing predictions
T_start = time.time()
y_pred = elm.predict(Test_X)
T_stop = time.time()


######################################################### Results For Classification training and testing ####################################################

print('******************Classification Training Results*******************')
print(f"Training time: {stop - start}m")
print ("training accuracy: ", accuracy_score(train_Y, tra_acc))

print("****************Testing Results*********************")
print(f"Testing time: {T_stop - T_start}m")
print ("testing accuracy: ", accuracy_score(test_Y, y_pred))

