########## >>>>>> Jordan Driscoll 905812

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
import random
sys.path.append("..")



# This initializes the Logistic Regression class 
class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

    #This is the vectorized sigmoid function 
    def _vectorized_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
        # takes the _sigmoid function and vectorizes it 
        vmoid = np.vectorize(LogisticRegression._sigmoid)
        #outputs a vectorized object 
        s_out = vmoid(s)
   
        
        return s_out
            
        # Hint: use the np.vectorize API

        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''
        #return 1 / (1 + e^-s)

        
        #e = math.e ** -s
        e = np.exp(-s)
        return (1 / (1 + e))
        
    # X --> Bias column is pre-inserted
    def _Batch_Gradient_Descent(X, y, lam, eta, iterations):
        #n - rows
        # d- cols 
        n, d = np.shape(X)
        
        # Initialize w 
        w = np.zeros((d,1))
        
        # the constant which is being multiplied unto the final result to average the result 
        c = eta / n
        
        # the reguralization constant 
        r = 1 - (2 * lam * c)
        
        for i in range(iterations):
            
            s = y * (X @ w)
            
            vs = LogisticRegression._vectorized_sigmoid(-s)
    
            # does the w equation 
            w = r * w  + (c * (X.T @ (y * vs)))

        return w
    
    # Code for teh Stochastic Gradient Descent
    def _Stochastic_Gradient_Descent(X, y, lam, eta, iterations, mini_batch_size):
        # should it get shuffled? 
        
        shuffle = True
        # n - rows
        # d - cols 
        n, d = np.shape(X)
        
        # Create boundries for the mini-batch-size
        if(mini_batch_size > n or mini_batch_size < 1):
            mini_batch_size = n
        
        m = mini_batch_size
        
        
        if(shuffle):
            # Makes it such that X & y are shuffled with each other 
            Xy = np.append(X, y, axis=1)
            # shuffles the Xy array 
            np.random.shuffle(Xy)
                
            # extracts y 
            y = Xy[:, d:]
                
            # extracts X 
            X = Xy[:, :d]
            
        
        # Initialize w 
        w = np.zeros((d,1))
        
        # the constant which is being multiplied unto the final result to average the result 
        c = eta / m
        
        # the reguralization constant 
        r = 1 - (2 * lam * c)
        
        curr_start = 0 
        curr_end = curr_start + m 
        start = curr_start 
        for i in range(iterations):

            #Create the miniature versions of the grpah 
            X_mini = X[start:curr_end , :]
            y_mini = y[start:curr_end , :]
            
            s = y_mini * (X_mini @ w)
            
            vs = LogisticRegression._vectorized_sigmoid(-s)
              
            
            
            # does the w equation 
            w = (c * ((y_mini * vs).T @ X_mini)).T + (r * w)
            
            if(n != m):
                    
                # update the current starting location
                curr_start += m
                # update end 
                curr_end = curr_start+m
                # set s to c 
                start = curr_start
        
                # if e's too big but c isn't 
                if(curr_end >= n and curr_start < n):
                    start = curr_start
                    curr_start = -m
                    curr_end = n
        
                # if e is too big 
                if(curr_end > n):
                    curr_start = 0
                    curr_end = curr_start + m
    
            
           
        
        # Sets the w to the result 
        return w 
        
        
        
        
        
        
        
    
    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 10, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''
        
        # Initialize the class-level degree to the inputted degree
        self.degree = degree
    
        
        
        # First, fix up X to put it into the correct degree
        X = MyUtils.z_transform(X, degree = self.degree)

        # Then add the bias column onto X 
        X = np.insert(X, 0, 1, axis=1)
        
        if (not SGD):
            self.w = LogisticRegression._Batch_Gradient_Descent(X, y, lam, eta, iterations) 
        else: 
            self.w = LogisticRegression._Stochastic_Gradient_Descent(X, y, lam, eta, iterations, mini_batch_size)
        
        
        
   
    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        # First, fix up X to put it into the correct degree
        X = MyUtils.z_transform(X, degree = self.degree)

        # Then add the bias column onto X 
        X = np.insert(X, 0, 1, axis=1)
        
        return LogisticRegression._vectorized_sigmoid(X @ self.w)
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        err = self.predict(X)
        error_total = 0 
        # Goes through each prediction
        for e1, y_i in zip(err, y):
            # Gives the error a label 
            e1_label = 1 if (e1 > .5) else -1 
            
            if(e1_label != y_i):
                error_total += 1  
        
        # Find all error
        
        return error_total
        

    
    

    
        
        
        
        

    
