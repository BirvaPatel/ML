## This is the code to calculate the Regression Accuracy.
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler         
from sklearn.model_selection import train_test_split
from elm import ELM 
from sklearn.metrics import accuracy_score, mean_squared_error
import time
from math import sqrt

####################################################Loading the Dataset ##########################################

column=['DATE','TIME','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT','PT08_S2_NMHC','NOX_GT','PT08_S3_NOX','NO2_GT','PT08_S4_NO2','PT08_S5_O3','T','RH','AH']
list_col=list(np.arange(len(column)))
data_set=pd.read_csv('AirQualityUCI.csv',header=None,skiprows=1,names=column,na_filter=True,na_values=-200,usecols=list_col)
print(data_set.head())
data_set.dropna(how='all',inplace=True)
data_set.dropna(thresh=10,axis=0,inplace=True)
print("size of main data",data_set.shape)
data_set['HOUR']=data_set['TIME'].apply(lambda x: int(x.split(':')[0]))
#print('Count of missing values:\n',data_set.shape[0]-data_set.count())
data_set['DATE']=pd.to_datetime(data_set.DATE, format='%d-%m-%Y')
data_set.set_index('DATE',inplace=True)
data_set['MONTH']=data_set.index.month     
data_set.reset_index(inplace=True)
data_set.drop('NMHC_GT',axis=1,inplace=True)    
data_set['CO_GT']=data_set['CO_GT'].fillna(data_set.groupby(['MONTH','HOUR'])['CO_GT'].transform('mean'))
data_set['NOX_GT']=data_set['NOX_GT'].fillna(data_set.groupby(['MONTH','HOUR'])['NOX_GT'].transform('mean'))
data_set['NO2_GT']=data_set['NO2_GT'].fillna(data_set.groupby(['MONTH','HOUR'])['NO2_GT'].transform('mean'))
#print('Left out missing value:',data_set.shape[0]-data_set.count())
data_set['CO_GT']=data_set['CO_GT'].fillna(data_set.groupby(['HOUR'])['CO_GT'].transform('mean'))
data_set['NOX_GT']=data_set['NOX_GT'].fillna(data_set.groupby(['HOUR'])['NOX_GT'].transform('mean'))
data_set['NO2_GT']=data_set['NO2_GT'].fillna(data_set.groupby(['HOUR'])['NO2_GT'].transform('mean'))
new_column=data_set.columns.tolist()[2:]
# Final X features and Y labels distribution. 
X=data_set[new_column].drop('RH',1)     
y=data_set['RH'] 

ss=StandardScaler()     
X_std=ss.fit_transform(X)     # stardardization
X_train, X_test, y_train, y_test=train_test_split(X_std,y,test_size=0.3, random_state=42)
print("size after cleaning and adjusting data")
print(X_train.shape)
print(X_test.shape)
################################################ Model implementation ########################################################


elm = ELM(200)
elm.fit(X_train, y_train)
# training predictions
start = time.time()	
tra_acc = elm.predict(X_train)
stop = time.time()
	
#testing predictions
T_start = time.time()
y_pred = elm.predict(X_test)
T_stop = time.time()


############################################### Results For Regression Training and Testing  #################################################

print('******************Training Results*******************')

print(f"Training time: {stop - start}m")
print ("training accuracy: ", accuracy_score(y_train, tra_acc))
train_rmse = sqrt(mean_squared_error(y_train, tra_acc))
print("training RMSE: ",train_rmse)

print("****************Testing Results*********************")

print(f"Testing time: {T_stop - T_start}m")
print ("testing accuracy: ", accuracy_score(y_test, y_pred))
test_rmse = sqrt(mean_squared_error(y_test, y_pred))
print("testing RMSE: ",test_rmse)




