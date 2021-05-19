from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2
from keras.optimizers import Adam,SGD
import os,keras
import time
#######################################  Preprocessing of data #########################################
# load the dataset
path = '15-Scene' 
categories = sorted(os.listdir(path))
ncategories = len(categories)

# Dividing First 100 images are considered as Training and rest as testing
data_train = []
labels_train = []
data_test = []
labels_test = []

for i, category in enumerate(categories):
    counter = 0;
	#np.random.shuffle(categories)
    for f in os.listdir(path + "/" + category):
        ext = os.path.splitext(f)[1]
        fullpath = os.path.join(path + "/" + category, f)
        label = fullpath.split(os.path.sep)[-2]
        image = cv2.imread(fullpath)
        image = cv2.resize(image, (224, 224))
        counter = counter + 1
        if (counter <= 100):      
            data_train.append(image)
            labels_train.append(label)
        else:
            data_test.append(image)
            labels_test.append(label)
            
print ('First 100 images per class are considered as Training and rest as testing')
# Normalizing the data
x_train = np.array(data_train, dtype="float") / 255.0
x_test = np.array(data_test, dtype="float") / 255.0
del data_train
del data_test
# LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(labels_train)
y_test = lb.fit_transform(labels_test)
del labels_train
del labels_test
print("Data Splitted")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
############################################ Model Implementation ############################################################
# Data augmentation
gen = ImageDataGenerator(width_shift_range=.2, # image augmentation using kearas imagedatagenerator function
                             height_shift_range=.2,
                          zoom_range=0.2)

test_gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=64)
test_generator = test_gen.flow(x_test, y_test, batch_size=64)
# Alexnet model with 5 convolution layer, 2 fully connected layer and 1 output layer.
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='softmax')
])

start = time.time()
adam = keras.optimizers.Adam(lr=0.01)
sgd = keras.optimizers.SGD(lr=0.01,momentum=0.9)
# compile the model to make a use of categorical cross-entropy loss function and sgd optimizer to get high accuracy.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# fitiing the model.
model.fit(train_generator,verbose=1,epochs=150,shuffle=True)   
##################################################### Results ################################################
score = model.evaluate(test_generator)
stop = time.time()
print(f"Training accuracy time with Alexnet model from scratch with Scene-15 : {stop - start}s")
print('Test accuracy:', (score[1]*100))

