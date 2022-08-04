##### >>>>>> Please put your name and 6-digit EWUID here


# Various tools for data manipulation. 



import numpy as np
import math

class MyUtils:

    
    def z_transform(X, degree = 2):
        
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)

        '''
        
        r = degree
        
        # degree is equal to 1, then return x 
        if r <= 1:
            return X
        
        n,d = X.shape()
        
        # Z is going to be a copy of x
        Z = X.copy()
        
        
        
        # next it is necessary to create all of the buckets
        # the # of buckets is conceptuall known d -r -1 C d - 1 
        # let's save those numbers in an array 
        
        #there will b r buckets 
        
        # B is a list with a bunch of buckets 
        B = []
        
        # the number of buckets 
        for i in range(r):
            # append a number - the ith bucket size which can be calculated w/ this equation
            # math.comb = n choose k 
            m = d+i # 0-based indexing t.f. the -1 is gone, d is the size of the X matrix 
            k = d-1 
            B.append(math.comb(m,k))
         
        ell = np.arange(np.sum(B)) # The summation of all the elements in the B array
        
        q = 0 # the total size of all of the buckets before the previous bucket
        
        p = d # the size of the previous bucket
        
        # at the beginning, there is one bucket 
        for i in range(1:r): # 1, 2, 3, ... r-1 range must t.f. be exclusive
            # create each bucket up to the ith bucket, visit the previous bucket 
            
            # go through every element in the previous bucket - the range starting from q going to q+p 
            for j in range(q, q+p):
                head = ell[j]
                # go from head to highest lexographically feature
                for k in range(head, d):
                    #elementwise multiplication
                    temp = (Z[: ,j] * X[:, k]).reshape(-1,1)
                    # insert new column temp on right side
                    Z = np.append(Z, temp, axis=1)
                    # j is hte index of the column you are currently computing
                    ell[j] = k # just multiplied w/ x's k column
                # adding previous bucket into p the new previous buck
                q += p 
                # the new previous bucket is going to be i which is the current i but will soon be updated 
                p = B[i]
                
        
        assert Z.shape[1] == np.sum(B)
        

                     
                    
                   
                    
                    
                
            
        
        
        
        return X

print("Hello World")

    
    