## Problem:

Similar to problem provided in assignment 1, utilizing deep learning (DL) techniques. The time required (in minutes) is provided for doctors
to attend patients at the ER at the hospital. Each doctor's expected time to attend each patient and the overall optimized value is provided in the dataset. The dataset
(folder name: DS.rar) contains 1000 .csv files, each illustrating the time required for 50 doctors and 50 patients. </br>
By extending the task from assignment 1, you have to apply deep learning techniques to train your models using the training data and then, in the test phase, predict the optimized time
required using the trained model.Write a python program to perform the following tasks:
1.Pre-process the dataset by properly loading the features/attributes and target variable into the training and testing part. Apply any feature scaling technique to scale the 
features before passing into the DL models.
2.Design an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) structure to predict the optimized value. Train the models using a cross-validation technique.
3.After training the models, save the trained models using a file named "ANN.h5" and "CNN.h5". Load the trained models from the saved files to predict 
the optimized value for the test data.
4.Visual predictive performance: Display the True vs. Prediction graph for both ANN and CNN models (one graph for each model).
5.Compare both models' prediction performance (ANN and CNN) using appropriate performance metrics.

## Solution:

#### Prediction Graph for CNN:</br> 
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-2/CNN.png)</br>
#### Prediction Graph for ANN:</br> 
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-2/ANN.png)</br>

#### Result:</br>
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-2/Perfomance_metrices.png)
