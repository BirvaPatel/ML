import numpy as np
import timeit

def matmul(a, b, k, iter):
	
	B = np.random.randint(35, size=(b,k))
	#print('matrix-B', B)
	for i in range(iter):

		A = np.random.randint(35, size=(a,b))
		#print('matrix-A', A)
		
		Result = np.matmul(A, B)
		'''
		res = np.zeros((a,k))
		# explicit for loops 
		for i in range(len(A)): 
			for j in range(len(B[0])): 
				for k in range(len(B)): 
					# resulted matrix 
					res[i][j] += A[i][k] * B[k][j] 
		print("one loop done")
		'''

start = timeit.default_timer()		
#condition-1
iter = 100
a = 500
b = 500
k = 400
matmul(a, b, k, iter)
stop = timeit.default_timer()
print('Time-1: %.1f ms'% (1000 * (stop - start) ) )

start = timeit.default_timer()
#condition-2		
iter1 = 5000
a1 = 50
b1 = 20
k1 = 50
matmul(a1, b1, k1, iter1)
stop = timeit.default_timer()
print('Time-2: %.1f ms'% (1000 * (stop - start) ) ) 

start = timeit.default_timer()
#condition-3
iter2 = 1000
a2 = 6
b2 = 4000	
k2 = 9
matmul(a2, b2, k2, iter2)
stop = timeit.default_timer()
print('Time-3: %.1f ms'% (1000 * (stop - start) ) )


	


