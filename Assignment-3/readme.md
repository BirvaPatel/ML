## Problem:
“DS.rar” contains a dataset of 1500 CT scans of various parts of the body such as abdomen, chest and head. You are required to train machine learning models in order to classify 
the provided images.
Write a python program to perform the following tasks:
1. Load the dataset for all three classes and resize each image to (32 x 32). Apply the required pre-processing steps to employ the data into Machine Learning / Deep Learning algorithms.
2. Construct a Convolutional Neural Network (CNN) architecture from scratch to extract features from the images. (HINT: Extract features for train and test set separately. 
Extract the features constructed by the convolutional layers from an intermediate dense layer. Please refrain from using any pre-trained model for implementing this step)
3. Apply the K-Nearest Neighbor (KNN) algorithm to the extracted features from CNN and find the optimal value of K. The value of K can be considered as [3, 5, 7, 9]. 
Determine the performance of the model using an appropriate performance metric. Draw a graph of K values and their corresponding performance in order to represent your results.
4. Apply Random Forest (RF) algorithm to the extracted features from CNN. Tune at least two hyperparameters using random search. Determine the model's optimal performance, the confusion matrix, and the value of hyperparameters producing the optimal performance.
5. Report the performance of each model and explain your results. (eg. overfitting, underfitting, etc.)

## Results:

For KNN, following all the parameters mentioned above, optimal value is at 3. Image below shows the performance result of KNN  
I generated the graph of my performance accuracy on different k values which shows that value of k is 3 and maximum accuracy is 0.89.</br>
Above mentioned diagram shows different accuracy on provided k value. For K=3, It is 0.89 which decreases significantly towards 5(0.875) then slight
increase by 7 that is 0.877 and the last one goes to upwards with 0.881 with k value of 9. </br>
So now it explains that when we increase the value of k it gives better result but when it reaches certain point then it starts to decrease which simply
means it experiences underfitting problem.</br>
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-3/graph.PNG)</br>

For RF, Used following Common parameters used in RF:</br>
#### n_estimators = Number of trees in random forest </br>
Usually I thought that, higher the number of trees the better to learn the data. 
However, adding a lot of trees can slow down the training process considerably. Which resulted in, increasing the number of trees decreases the test performance. 
That is why I needed to find the exact spot which can be find by random search method. 
</br>I took following random values at first: ➔ [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] </br>
#### max_features = Number of features to consider when looking for the best split. </br>
The moment I tried to increase the features, it was fitting perfectly to training ser but the worst for teasting set so, this is an overfitting case. 
For this, I selected following random values: ➔ ['auto', 'sqrt'] </br> 
#### max_depth = Maximum number of levels in tree</br>
As the name suggests, the deeper the tree, the more splits it has, and it captures more information about the data. 
But my model overfits for large depth values. The trees perfectly predict all of the train data, however, it fails to generalize the findings for new data.
Random values for this are following: ➔ [int, None] </br>
#### min_samples_split = Minimum number of samples required to split a node </br>
Here I tried to vary the parameter from 10% to 100% of the samples I found that when I require all of the samples at each node, the model cannot learn enough about the data.
This is an underfitting case. ➔ [2, 5, 10] </br>
#### min_samples_leaf = Minimum number of samples required at each leaf node </br>
This parameter is similar to min_samples_splits, 
however, this describe the minimum number of samples at the leafs, the base of the tree. My experiment showed that Increasing this value, 
caused underfitting. ➔ [1, 2, 4] </br>
#### Bootstrap = Method of selecting samples for training each tree</br> 
➔ [True, False] </br>
• used random grid to search for best hyperparameters • from sklearn.model_selection import RandomizedSearchCV.</br>
• then using 3 fold cross validation, search across 100 different combinations, and use all available cores, finally fit the model in rf_random.


