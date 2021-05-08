import numpy as np

A = np.array([[1,1/2,1/3],[1/3,1/2,1]])
B = np.array([[0.5,1,6],[3,-4,2]])

hadamard = np.multiply(A,B)
print(hadamard)

AB_T = np.matmul(A, B.T)
print(AB_T)

BA_T = np.matmul(B, A.T)
print(BA_T)

# x = np.array([1,0,1,0]).T
# w = np.array([5,4,6,1]).T
# w_t_x = np.matmul(w.T, x.T)
# print(w_t_x)
# print(11**2)