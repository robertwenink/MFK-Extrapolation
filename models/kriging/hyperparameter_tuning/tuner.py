"""
Superclass for the hyperparameter tuning process, providing basic functionality
"""

class Tuner:

    def __init__(self,d):
        self.Theta = np.zeros(d)
        
    def getHyperParams():
        return self.Theta


    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv
    # a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])

   #  inv(a)
    # array([[[-2.  ,  1.  ],
    #         [ 1.5 , -0.5 ]],
    #        [[-1.25,  0.75],
    #         [ 0.75, -0.25]]])