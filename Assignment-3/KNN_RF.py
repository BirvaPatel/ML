import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sn
from keras import *
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam,SGD
from keras.preprocessing import image
from sklearn import metrics
from skimage import io
import warnings
warnings.simplefilter(action='error', category=FutureWarning)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

################################ Pre-Processing ############################################################################
#Add the path to the dataset
path = r'DS/' 
# array x stores the feature images
X=[] 
#array labels stores 1500 labels
labels=[]
#finds the label of images from given path name(to folder name)of the directory.
for label in os.listdir(path): 
	back_path = os.path.join(path,label)
	# take folder name and split it by '_'.so that we can store our labels in labels array.
	labels.append(label.split('_')[0]) 
# this for loop takes all images from directory.
for filename in os.listdir(path):  
	#check for only images.
	if not os.listdir(path)[-1].endswith('.jpeg'): 
		continue
	image_path = os.path.join(path,filename)
	# load_img function to load image with target size of 32*32.
	img = image.load_img(image_path,target_size=(32,32),color_mode = 'grayscale') 
	# img_to_array function to convert image to array which will give array 128*128*3.
	img = image.img_to_array(img) 
	# normalize with highest pixelvalue.
	img /= 255
	# ready images are stored in X array.
	X.append(img) 
#print the unique label name from labels array.
print(np.unique(labels))
# data split by slicing(70:30)-- from each class(500)=(300train)+(200test)
#for X, 900 training samples and 600 testing samples from each class.
x_train = X[:300] + X[500:800] + X[1000:1300]	
x_test = X[300:500] + X[800:1000] + X[1300:1500]
#for Y-labels, 900 training samples and 600 testing samples from each class.
y_train = labels[:300] + labels[500:800] + labels[1000:1300]	
y_test = labels[300:500] + labels[800:1000] + labels[1300:1500]
#converting them to numpy array.
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#label binarizer to use in random forest classification.
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_y_train = encoder.fit_transform(y_train)
transfomed_y_test = encoder.fit_transform(y_test)


#defining the sequential model.
model=Sequential()
#first convolution layer with input shape of 32*32*1.
model.add(Conv2D(32,(3,3),input_shape=(32,32,1)))
#apply relu activation 
model.add(Activation('relu'))
#maxpooling layer-1
model.add(MaxPooling2D(pool_size=(2,2)))
#convolution layer-2
model.add(Conv2D(64,(3,3)))
#apply relu activation
model.add(Activation('relu'))
#Maxpooling layer-2
model.add(MaxPooling2D(pool_size=(2,2)))
#fatten layer
model.add(Flatten())
#dense layer-1
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#dense layer-2
model.add(Dense(3))
#final softmax activate 
model.add(Activation('softmax'))
model.summary()
#compile the model with adam optimizer.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

###################################### feature extraction #############################################################

#extracting features from feature layer of cnn model.
new_model=Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
#predict new train_x for future classification.
train_x=new_model.predict(x_train)
#predict new test_x for future classification.
test_x=new_model.predict(x_test)

################################################ K Nearest Neighbours Classifier######################################################
#importing necessary library for KNN.
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#array of different accuracy for different k.
acc = []
#list of K values.
k=[3,5,7,9]
#for loop to find optimal k for maximum accuracy.
for i in k:
	knn_model = KNeighborsClassifier(n_neighbors = i).fit(train_x,y_train)
	y_pred = knn_model.predict(test_x)
	acc.append(metrics.accuracy_score(y_test, y_pred))
#taking the index of highest accuracy.	
optimal_k = k[acc.index(max(acc))]	
print('___________________________________________KNN_______________________________________________________________________________')
print("Maximum accuracy:-",max(acc),"at K =",optimal_k)
knn = KNeighborsClassifier(n_neighbors = optimal_k).fit(train_x,y_train)
#pickle file to save and load model.
Pkl_Filename = "KNN.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(knn, file)
with open(Pkl_Filename, 'rb') as file:  
    KNN_Model = pickle.load(file)
#new predicted y from loading the model from pickle.just to verify the loaded model.
y_knn = KNN_Model.predict(test_x)
#graph for accuracy respective of different values of k.
plt.figure(figsize=(10,6))
plt.plot(k,acc,color = 'red',linestyle='solid',marker='o',markerfacecolor='black', markersize=11)
plt.title('Accuracy For different K-Values')
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.show()
print('**********************************************************')
print('here is a confusion matrix')
print(confusion_matrix(y_test, y_knn))
print('**********************************************************')
print(classification_report(y_test, y_knn))

#################################### RandomForestClassifier #############################################################################
print('___________________________________________RF-- will take time to execute____________________________________________________')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],'max_features': ['auto', 'sqrt'], 'max_depth': [[int(x) for x in np.linspace(10, 110, num = 11)],None],
               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}

# used random grid to search for best hyperparameters
rf = RandomForestClassifier()
# using 3 fold cross validation, search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_x,transfomed_y_train)

Pkl_Filename = "RF.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random, file)
with open(Pkl_Filename, 'rb') as file:  
    RF_Model = pickle.load(file)

Best_RF = rf_random.best_estimator_
#Best_RF.fit(train_x,transfomed_y_train)
y_pred = RF_Model.predict(test_x)

Y_pred = np.argmax(y_pred, axis=1)
Y_test=np.argmax(transfomed_y_test, axis=1)
print('**********************************************************')
print('Value of hyperparameters producing the optimal performance:',rf_random.best_params_)
print('**********************************************************')
print("Accuracy: {0}".format(accuracy_score(Y_test, Y_pred)))
print('**********************************************************')
print(confusion_matrix(Y_test, Y_pred))
print('**********************************************************')
print(classification_report(Y_test, Y_pred))

