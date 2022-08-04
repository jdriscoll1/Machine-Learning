###### Jordan Driscoll - 905812    <<<<<<<<<<<



class Rectangle:
     
    # add the init function here that creates and initalize the
    # attributes of an object of this class. The attributes are
    # named as "length" and "width". The init function takes two
    # arguments that are to assigned to the length and width

    # write the init function here
    def __init__(self, length, width):
        #Initialize Length & Width
        self.length = length 
        self.width = width

    
    
    # add another function here called "modify" that takes two arguments that
    # are to be assigned to the two attributes of the object.
    def modify(self, length, width):
        #Update Length & Width
        self.width = width
        self.length = length
        

    
    
    
    # add annother function here called "area" that is to compute
    # and return the area of the rectangle. 
    def area(self):
        #Area function
        return self.width * self.length

    
    
    

    # add annother function here called "perimeter" that is to compute
    # and return the perimeter of the rectangle. 
    def perimeter(self):
        #Perimeter Function
        return 2 * (self.length + self.width)
