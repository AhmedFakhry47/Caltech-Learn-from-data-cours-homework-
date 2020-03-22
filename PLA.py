'''
A simple code (Script) to illustrate the idea behind perception 
learning algorithm
'''
import numpy as np

##First step is generating data
D = np.random.rand(3,1000)
threshold = .4 


#target function represented in weights 
f  	    =  np.array([0.2,0.8,0.5]).reshape((3,1))
Y  	    = (np.sum((f*D),axis=0) > threshold).astype(int) 
Y[Y==0] = -1


##Start Applying PLA 
W  = np.zeros(3)	# Initialization of weights by zeros
h  = 0 					# The current hyposis 
N  = 1000				# Number of training examples 

#Start learning 
for _ in range(10000):
	for i in range(N):
		h = 1 if (np.sum(W * D[:,i]) > .4) else -1
		if (h != Y[i]):	#Update step in case of miss-classified points 
			W+= Y[i]*D[:,i]















