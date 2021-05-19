import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ELM (BaseEstimator, ClassifierMixin):

	def __init__(self,max_hid_num,a=1):
		self.max_hid_num = max_hid_num
		self.a = a

	def _sigmoid(self, x):
		
		sigmoid_range = 34.538776394910684
		x = np.clip(x, -sigmoid_range, sigmoid_range)
		return 1 / (1 + np.exp(-self.a * x))

	def _add_bias(self, X):

		return np.c_[X, np.ones(X.shape[0])]

	def _ltov(self, n, label):
		
		return [-1 if i != label else 1 for i in range(1, n + 1)]

	def fit(self, X, y):
		
		# number of class, number of output neuron
		self.out_num = 10
		self.hid_num = 1
		self.min_error = 0
		self.E = []
	
		while self.hid_num<self.max_hid_num is True and self.E<= self.min_error is True:
			self.hid_num = self.hid_num+1
			# calculating bias to feature vectors
			X = self._add_bias(X)
			# weights between input layer and hidden layer
			np.random.seed()
			self.W = np.random.uniform(-1., 1.,(self.hid_num, X.shape[1]))
			print(self.W.shape)
			# find inverse weight matrix
			_H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))
			# weights between hidden layer to output layer
			self.beta = np.dot(_H.T, y)  
			self.E = self.E- np.dot(self._sigmoid(np.dot(self.W, X.T)),self.beta)
			
		return self
	
	def predict(self, X):
		
		_H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
		y = np.dot(_H.T, self.beta)
		return np.sign(y)
		

