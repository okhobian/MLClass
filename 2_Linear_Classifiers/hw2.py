import numpy as np 
import matplotlib.pyplot as plt

import numpy # import again 
import matplotlib.pyplot # import again 

import numpy.linalg 
import numpy.random

import hashlib

def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class, 
      keys are mx (center on x axis), my (center on y axis), 
               ux (sigma on x axis), ux (sigma on y axis), 
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']), 
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']), 
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack(( Para1['y']*numpy.ones(Para1['N']), 
                       Para2['y']*numpy.ones(Para2['N'])  ))            
    X = numpy.hstack((X1, X2)) 
    X = numpy.transpose(X)
    return X, Y 

def plot_data_hyperplane(X, y, w, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array, the labels 
    w: 1-by-3 numpy array, the last element of which is the bias term

    Examples
    --------------

    >>> X = numpy.array([[1,2,3], \
                         [4,5,6], \
                         [7,8,9]]) 
    >>> y = numpy.array([1,-1,1])
    >>> w = [1, 2, -10]
    >>> filename = "test.png"
    >>> plot_data_hyperplane(X, y, w, filename)
    >>> hashlib.md5(open(filename, 'rb').read()).hexdigest()
    '9df878adf0a4f5276ade67e61d00a123'
    """

    # your code here
    
    # separte two classes
    X1 = X[y == +1]
    X2 = X[y == -1]
        
    # plot data samples
    plt.plot(X1[:,0], X1[:,1], 'ro')
    plt.plot(X2[:,0], X2[:,1], 'bo')
    
    # plot line
    x_ticks = np.array([np.min(X[:,0]), np.max(X[:,0])])
    y_ticks = -1*(x_ticks * w[0] + w[2])/w[1]
    plt.plot(x_ticks, y_ticks, '-')
    
    # set limit
    plt.xlim(np.min(X[:,0]), np.max(X[:,0]))
    plt.ylim(np.min(X[:,1]), np.max(X[:,1]))
    
    # save, close
    plt.savefig(filename)
    plt.close('all')

def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_mse(X, y, 'test1.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    >>> X,y = generate_data(\
    {'mx':1,'my':-2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
    {'mx':-1,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
    seed=10)
    >>> # print (X, y)
    >>> plot_mse(X, y, 'test2.png')
    array([ 0.93061084, -0.01833983,  0.01127093])
    """
    w = np.array([0,0,0]) # just a placeholder

    # your code here
    
    # convert X into augmented
    X = np.hstack((X, np.ones(len(X)).reshape(len(X),1)))
    
    # w = (X^T X)^-1 X^T y
    compound = np.matmul(numpy.transpose(X), X)
    all_but_y = np.matmul(np.linalg.inv(compound), numpy.transpose(X))
    w = np.matmul(all_but_y, y)

    # Plot after you have w. 
    plot_data_hyperplane(X, y, w, filename)

    return w

def plot_fisher(X, y, filename): 
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_fisher(X, y, 'test3.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
        {'mx':-1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
        {'mx':2,'my':-4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=1)
    >>> plot_fisher(X, y, 'test4.png')
    array([-1.54593468,  0.00366625,  0.40890079])
    """

    w = np.array([0,0,0]) # just a placeholder

    # your code here 

    # separte two classes
    # X1 = X[y == +1].T   # make into one sample/colum
    # X2 = X[y == -1].T
    X1 = X[y == +1]       # make into one sample/row
    X2 = X[y == -1]

    # compute c_i
    c1 = np.count_nonzero(y == +1)
    c2 = np.count_nonzero(y == -1)
        
    # compute m_i
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)

    # compute X_i
    X1 = X1.T
    X2 = X2.T
        
    # compute M_i
    M1 = np.array([m1]*c1).T
    M2 = np.array([m2]*c2).T
       
    # compute S_i
    XminusM = X1 - M1
    S1 = np.matmul(XminusM, XminusM.T)
    XminusM = X2 - M2
    S2 = np.matmul(XminusM, XminusM.T)
        
    # compute S_w
    Sw = S1 + S2
    
    # compute w
    w = np.matmul(Sw.T, np.array(m1-m2))
    w = np.hstack((w,np.ones(1)))

    # Plot after you have w. 
    plot_data_hyperplane(X, y, w, filename)
    return w


if __name__ == "__main__":
    import doctest
    doctest.testmod()