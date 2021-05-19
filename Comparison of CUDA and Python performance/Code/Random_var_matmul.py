import numpy as np
import timeit

def matmul(a, b, k, iter):

	for i in range(iter):

		A = np.random.uniform(low=0.0, high=1.0, size= (a,b))
		#print('matrix-A', A)
		B = np.random.uniform(low=0.0, high=1.0, size= (b,k))
		#print('matrix-B', B)
		result=np.matmul(A,B)  
		#print('final result', result)
		

start = timeit.default_timer()		
#condition-1 but this time with 1000 iterations
iter = 200
a = 5000
b = 5000
k = 4000
matmul(a, b, k, iter)
stop = timeit.default_timer()
print('Time to run on CPU: %.1f ms'% (1000 * (stop - start) ) ) 
