# >>>>>>>>>>>>> Please place your full name and 6-digit EWU ID here
 
# Implementation of the perceptron learning algorithm. Support the pocket version for linearly unseparatable data.
 
 
 
# NOTE: Your code need to vectorize any operation that can be vectorized.
 
import numpy as np
# this class will provide this service
# if you want to get a binary classifier, you give me some training set & its labels
# the class does the learning & training and figures out the clssifier (w-vecotr)
# onc ethe training is done you get a new sample, and the label will take the sample and return the model
# it'll take a bunch of new samples and give a bunch of w-vectors back k
# and also the third service is that once hte model is trained then the company takes a new data set it has all the sample features and all hte labels and the business wants to test if it's doing a goo d job
# it shifts the new data set to the model and use the w-vector to predict the label with every sample and compare it w/ the business provided sample and compare how many are matched
# it will return the number of misclassified sample
 
# 1) Training
# 2) After Training: Future Sample and Tell the Label
# 3) New Data Set whose label is known nad you want ot check if ml is doing good job or not ship it and it'll say
 
class PLA:
    #What is the information you need to maintain in your ML product?
    # w --> once stored you will know the number of features every sample will provide because w has d+1 sample size
    def __init__(self):
        self.w = None
   
    # takes several parameters
    # 1) self --> The class
    # 2) X --> The Big Training Sample Matrix, provided by user of model. Does not include the bias feature column. The user should not know & should not care
    # 3) y --> Label vector, 2D array of +1 and -1 representing the label of the n-samples in X
    # 4) pocket --> It will run pocket version
    # 5) number of times to run pocket
 
    # take x add a new column to each element, 1
    # make x matrix wider by adding new column on the left adding the bias feature
    # initialize self.w to be 0
    # copy the code that he sends us here - do  some modification to handle the pocket versio n
    # pocket version continues running and recording hte best w
    def fit(self, X, y, pocket = True, epochs = 100):
        print("this is a test")
        return 1
        ''' X: n x d matrix, representing n samples, each has d features. It does not have the bias column.
            y: n x 1 matrix of {+1, -1}, representing the n labels of the n samples in X.
 
            return: self.w, the (d+1) x 1 weight matrix representing the classifier.  
 
            1) if pocket = True, the function will run the pocket PLA and will update the weight
               vector for no more than epoch times.  
 
            2) if pocket = False, the function will run until the classifier is found.
               If the classifier does not exist, it will run forever.            
            
           
        '''
       
        ### add your code here.

         # Add the bias feature to every instance of X --> This will be determined +1, -1, or 0 depending on if it is 

         # We of course need the equation that will allow the PLA algorithm to work 

         # This will 
       
        # Hint: In practice, one will scan through the entire set to find the next misclassied sample.
        #       One will repeatedly scan the data set until one scan does not see any misclassified sample.
       
        #       Use matrix/vector operation in checking each training sampling.
       
        #       In the pocket version, you can use the self.error function you will develop below.
        # first create a w that probably won't even be right
        # First, find if there is a misclassified sample
        # If there is a misclassified sample: follow these easy steps
          # X  - [1 or -1, x, y]
          # Look for a misclassified sample 
          # a) take and store that misclassified sample -- 3-Dimensional Coordinate From The X Matrix 
        
          # b) update w_i: w_i += d*x_i
          # ba) i --> the index of the misclassified value: are we updating w_1 or w_2 (or w_0?)
            # we might be able to figure this out by playing around with it. But if it's checking both then it feels like it should always be 1
          # bb) d --> 
            # bba) Case 1: 1 if too low
            # bbb) Case 2: -1 if too high
        # return the updated w

        #TODO Deal w/ epochs

      
    

    #---------------------------------------------------------------------------------------------------------------------------------------------
  
    # X is a big matrix
    # a collection of n samples
    # every row in matrix is a sample and does not include bias feature
    # this provides prediction for batch of new samples
    # it can do batch, it can also do just one sample
    # return +1, -1, or 0

    # X @ self.w --> New matrix tha tis n x 1 and everything there is a number where you do np.sign(to vector) which gives you new vector where everything is psotive negtive or 0
    def predict(self, X):
        ''' X: n x d matrix, representing n samples and each has d features, excluding the bias feature.
            return: n x 1 vector, representing the n labels of the n samples in X.
           
            Each label could be +1, -1, or 0.
           
            Note: We let the users to decide what to do with samples
                  that sit right on the classifier, i.e., x^T w = 0
        '''
 
        ### add your code here
       
        # Hint: use matrix/vector operation to predict the labels of all samples in one shot of code.
   
   
    # error - people give you a new data set including it's labels you take it's X and y values for label sy ou take it in and aedd bias feature columna dus e your w to produce your predicted label vector now you have your predicted y vector and hte given label vector and compare these two ys and see how many errors there are tand then uyou return that number to the user isf everything is good that is good if it's like 50 that means the model is very abd or hte data is chanign and you needa retrain the model bu thtat last part shouldn't be relevant in this particular thin gna now let me run ithis codc ehta tuses this class na dit should reall yreturn some w that is represetnting error
    # X --> the data
    # y --> The correct result
    # result --> # compare
    def error(self, X, y):
        ''' X: n x d matrix, representing n samples and each has d features, excluding the bias feature.
            y: n x 1 vector, representing the n labels of the n samples in X. Each label is +1 or -1.
           
            return: the number of samples in X that are misclassified by the classifier
           
            Note: we count a sample x that sits right on the classifier, x^T w = 0, as a misclassified one.
        '''
       
        # add your code here
       
        # Hint: use matrix/vector operation to get predicated label vector in one shot of code.
        #       Then use vector comparison to compare the given label vector and
        #       the predicted label vector, along with the help from the numpy.sum function
        #       to count the #misclassified quickly. 

# CREATE DATA SET
 
# 1 by 1 data set where every feature is a number frojm 0 to 1
# Randomly create artbitrary number of x y dots - those are sample locations
# Generate 3 random numbers for w0, w1, w2 - that is the target line being looked ofr
# Label the data - dots are random and use the randomly generated w vector to go through every dot which will either give a positive value, negative value or 0
 
# plug each dot into w0 + w1*x1 + w2 * x2 --> This will be postive, negative or 0
# create positive one label
# if the number is 0, then shift it a bit
 
# this will create a training set of samples
 
# ^^^
 
# Give training set to PLA and see if PLA can see target line
# Goal is for PLA to create cutting line that doens't _have_ to be intial w line but should be roughly close
 
# the more data the harder a time the pla will find it because the cluster will be much cloesr
 
# the real speed is sig faster than display speed because display is very slow :(
 
# w0 is only being updated by 1 every tgime and it will either be 1, or 0
 
#commands:
# w = w + (w[i]*x[i]).respahe(-1, 10

 
# 3 elements in array w0, w1, w2
# Reshape into a column vector [0, 0, 0].reshape(-1,1)
# updated that keeps track of
# clear output
 
#for every sample
# big matrix X every row is a training sample that has b + 1 features
# if(np.sighn(x[i] @ w 0 != y[i]
# @ --> Operator that  multiplies two matrices
# x[i] is a row vector
# w is a column vector
# sample is above or below current w
# np.sign and outputs if the number is positive or negative
# if they are not equal then it is misclassified and use the pla algorithm and update w
    # w = w + (y[i] * x[i]).reshape(-1, 1)
        #y[i] is a column vector it is an nx1 matrix
        #x[i] is a row vector
        # * --> element wise multiplication such that x[i]*y
        # w is a row vector and (y[i]*x[i]) is a column vector
        # reshpae parameter -1: the interpreter determines the value
 

