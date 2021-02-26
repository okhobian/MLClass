import numpy as np 
import numpy
import matplotlib.pyplot as plt

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
    """
    # separte two classes
    X1 = X[y == +1]
    X2 = X[y == -1]
    
    # plot data samples
    plt.plot(X1[:,0], X1[:,1], 'ro')
    plt.plot(X2[:,0], X2[:,1], 'bo')
    # plt.scatter(X1[:,0], X1[:,1], s=20, facecolors='none', edgecolors='r')
    # plt.scatter(X2[:,0], X2[:,1], s=20, facecolors='none', edgecolors='b')
    
    # plot line
    x_ticks = np.array([np.min(X[:,0]), np.max(X[:,0])])
    y_ticks = -1*(x_ticks * w[0] +w[2])/w[1]
    
    # set limit
    plt.xlim(np.min(X[:,0]), np.max(X[:,0]))
    plt.ylim(np.min(X[:,1]), np.max(X[:,1]))
    
    # plot, save, close
    plt.plot(x_ticks, y_ticks)
    plt.show()
    # plt.savefig(filename)
    plt.close('all')

def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array
    """
    w = np.array([0,0,0]) # just a placeholder

    # your code here

    # convert X into augmented
    X = np.hstack((X, np.ones(len(X)).reshape(len(X),1)))
    
    compound = np.matmul(numpy.transpose(X), X)
    all_but_y = np.matmul(np.linalg.inv(compound), numpy.transpose(X))
    w = np.matmul(all_but_y, y)

    # Plot after you have w. 
    plot_data_hyperplane(X, y, w, filename)

    return w


if __name__ == "__main__":

    X,y = generate_data(
            {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20},
            {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},
            seed=10)

    # w = [1, 2, -10]
    # plot_mse(X, y, "test.png")
    # plot_data_hyperplane(X, y, w, "test.png")


# X = np.array([[1,2],
#               [4,5],
#               [7,8]]) 
# y = np.array([1,-1,1])
# w = [1, 2, -10]

# X1 = X[y == +1]
# X2 = X[y == -1]
    
# plt.plot(X1[:,0], X1[:,1], 'ro')
# plt.plot(X2[:,0], X2[:,1], 'bo')
    
# x_ticks = np.array([np.min(X[:,0]), np.max(X[:,0])])
# y_ticks = -1*(x_ticks * w[0] +w[2])/w[1]
# plt.xlim(np.min(X[:,0]), np.max(X[:,0]))
# plt.ylim(np.min(X[:,1]), np.max(X[:,1]))
# plt.plot(x_ticks, y_ticks)
# # plt.savefig(filename)
# plt.show()
# plt.close('all')

# print(X)
# print(np.ones(len(X)))

# X = np.hstack((X, np.ones(len(X)).reshape(len(X),1)))
# print(np.ones(len(X)))
# print(X)