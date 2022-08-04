# Jordan Driscoll 905812





# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
import math_util as mu
import nn_layer


class NeuralNetwork:
    
    
   
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L.

    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        # Create the new neural layer using the passed in variables
        new_layer = NeuralLayer(d, act)
        # Add the layers to the list of layers
        self.layers.append(new_layer)
        # Add the layer index
        self.L += 1
        
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        
        weight_rng= np.random.default_rng(2142)
        # We go through the layers excluding the first one 
        for layer_id in range(1, self.L + 1): 
            
            curr_layer = self.layers[layer_id]
            prev_layer = self.layers[layer_id - 1]
           
            # go through and create the w for each l 
            layer_n = prev_layer.d + 1 # number of nodes this layer, +1 is included as bias  
            layer_d = self.layers[layer_id].d # number of connections this layer
            
            range_min = -1 / math.sqrt(layer_d)
            range_max = 1 / math.sqrt(layer_d)
            
            curr_layer.W = weight_rng.uniform(low=range_min, high=range_max, size=(layer_n,layer_d))
            
            # curr_layer.W = np.random.uniform(low=range_min, high=range_max, size=(layer_n,layer_d))
            # Create a Matrix of Zeros of Size l_n, l_d 

    
    
    ############### Update Final Delta and Gradient ############################
    def _update_final_gradient_and_delta(self, c, Y):

        final_layer = self.layers[self.L] 
        
        transform_str = "ij,ik->jk"
        
        # Delta^L --> Matrix of all of the vectors of all of the partial derivatives of the errors
        # D_L = 2 * (X* - Y) * derrived activation(S)            
        # The final X without the bias 
        X_star = final_layer.X[:, 1:]
        
        # The final S 
        S_final = final_layer.S
        
        # The final Delta
        a = 2 * (X_star - Y)
        b = final_layer.act_de(S_final)
        
        Delta_final = a * b
                
        final_layer.Delta = Delta_final
        
        ############### OBATINING G ###########################
        # G is the gradient --> This is the line of ups & downs
        # Second to last layer's X 
        A = self.layers[self.L - 1].X
        
        B = Delta_final
        
        # Update the final G
        final_layer.G = c * np.einsum(transform_str, A, B)

    
    def _backpropogate(self, layer_id, c):
        # The previous layer 
        p_layer = self.layers[layer_id - 1]
        
        # The Current Layer
        layer = self.layers[layer_id]
        
        #The next layer
        n_layer = self.layers[layer_id + 1]
        
        # Obtain the S with the derived activation function applied to it 
        S = layer.act_de(layer.S)
        
        # The W in the equation w/out bias 
        W = n_layer.W[1:].T 
        
        # The delta is the next layer's delta
        d = n_layer.Delta
        
        layer.Delta = S * (d @ W)
        
        # X is the previous layer's X
        A = p_layer.X
        
        B = layer.Delta
        
        transform_str = "ij,ik->jk"
        
        # Calculate Gradient from each layer 
        layer.G = c * np.einsum(transform_str, A, B)
    
    
    def _update_minibatch(self, d):
        # update c 
        d['c'] += d['m']
        # update end 
        d['e'] = d['c']+ d['m']
        # set s to c 
        d['s'] = d['c']
        if(d['n'] == d['m']):
            d['s'] = 0
            d['c'] = 0 
            d['e'] = d['n']
        else:
            # if e's too big but c isn't 
            if(d['e'] >= d['n'] and d['c'] <= d['n']):
                d['s'] = d['c']
                d['c'] = -d['m']
                d['e'] = d['n']
                
            # if e is too big 
            # e > n 
            if(d['e'] > d['n']):
                # c = 0
                d['c'] = 0
                # e = c + m
                d['e'] = d['c'] + d['m']
        
            return d
        
    # Prediction Should Be (97%)
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        
        shuffle = True
            
        # The first layer
        first_layer = self.layers[0]
        
        # Obtain the shape of n 
        n, d = X.shape
        
        if(not SGD):
            mini_batch_size = n        
        
        m = mini_batch_size
        
        # Shuffle X * y 
        if(shuffle):
            # Makes it such that X & y are shuffled with each other 
            Xy = np.append(X, Y, axis=1)
            # shuffles the Xy array 
            np.random.shuffle(Xy)
                
            # extracts y 
            Y = Xy[:, d:]
                
            # extracts X 
            X = Xy[:, :d]

        
        # This stores all of the necessary data for the minibatch
        minibatch_data = dict({'n':n, 'c':0, 'm':mini_batch_size,'e':mini_batch_size,'s':0})
        
        
        
        
        
        # Run for the number of times that are available
        for iteration in range(iterations):           
            
            X_mini = X[minibatch_data['c']:minibatch_data['e']]
            y_mini = Y[minibatch_data['c']:minibatch_data['e']]
            
            c = 1 / X_mini.shape[0]
            
            # Add Bias Column to the 0th layer's X 
            first_layer.X = np.insert(X_mini, 0, 1, axis=1)
        
            
            
            ######### FORWARD FEEDING #################
            # For Each Layer Excluding the First One 
            for layer_id in range(1, self.L + 1):       
                self._Forward_Feed(layer_id)    
                
     

            ############ UPDATE FINAL G & D #################
            # Update the final gradient and Delta
            self._update_final_gradient_and_delta(c, y_mini)

            

            
            ############## BACKPROPOGATION #######################
            # Then go through and update all the delta previous to that
            # Back Propogation: Starting from the back going to the front
            for layer_id in reversed(range(1, self.L)):
                self._backpropogate(layer_id, c)
                
                
            
            ############## UPDATE THE WEIGHTS ################
            for layer_id in range(1, self.L + 1):
                layer = self.layers[layer_id]
                layer.W -= eta * layer.G
         
            # Adjust 
            if(SGD):
                minibatch_data = self._update_minibatch(minibatch_data)
            
            
            
            
            
            
            
        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

        pass

    
    def _Forward_Feed(self, layer_id): 
        
        # the previous layer
        p_layer = self.layers[layer_id - 1]
            
        # the current layer 
        layer = self.layers[layer_id]
        
        # Set the S 
        S = p_layer.X @ layer.W
        
        # Run the activation function on S
        layer.X = layer.act(S)
        
        # Add bias to the X 
        layer.X = np.insert(layer.X, 0, 1, axis=1)
        
        # Set the layer's S 
        layer.S = S 
        

        
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        # Get X to the right shape 
        # Set as an np array
        X = np.array(X)
        
        X = X.reshape(-1, self.layers[0].d)
        
        # Add bias column
        X = np.insert(X, 0, 1, axis=1)
        
        # Take the first layer and make it's inputted X the sample X being predicted
        self.layers[0].X = X
        
        # go through layer excluding the first one 
        for layer_id in range(1, self.L + 1):
                       
            self._Forward_Feed(layer_id)
            
            
        
        # Take the final layer 
        final_layer = self.layers[self.L]
        
        # Knock off the bias column
        final_layer.X = final_layer.X[:, 1:]
        
        # Then return an n x 1 with the arg max's
        out = np.argmax(final_layer.X, axis=1)
        
        out = out.reshape(-1, 1)
        
        return out 
         
            
             
            
          
            
            
            
            
            
        
        return _Forward_Feed(X)
    
        
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
            
        '''
        
        n, d = X.shape
        
        # Find the nx1 predicition matrix 
        preds = self.predict(X)
        
        # Take all of the maximum's from the Y matrix - results in an nx1 matrix
        y = np.argmax(Y, axis=1) 
        
        y = y.reshape(-1, 1) 
        
        # Return the summation of the number of equal args
        err = sum(preds != y)
        
        # # of err / # of samples
        return err / n
   