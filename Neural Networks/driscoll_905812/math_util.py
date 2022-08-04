# Jordan Driscoll 905812




import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        v_tanh = np.vectorize(np.tanh)
        return v_tanh(x)
   

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        v_tanh_de = np.vectorize(lambda x: 1 - (np.tanh(x) ** 2))
        return v_tanh_de(x)

    
    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        v_sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        return v_sigmoid(x)
         
        
        

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        # The sigmoid function
       
        # The vectorized sigmoid function
        sigmoid = lambda s: 1 / (1 + np.exp(-s))
        
        v_sigmoid = np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x)))
        
        return v_sigmoid(x)
        

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return np.array(x) 
        

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all zeros of the same shape of x.
        '''
        return np.ones(np.array(x).shape)
        

    
    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        v_relu = np.vectorize(lambda x: x * (x > 0))
        return v_relu(x)
    
    

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        return 1 if x > 0 else 0 

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        v_relu_de = np.vectorize(MyMath._relu_de_scaler)
        return v_relu_de(x)
