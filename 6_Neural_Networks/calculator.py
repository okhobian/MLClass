import numpy as np

A = np.array([[1,2,3],[3,2,1]])
B = np.array([[0.5,0.1,0.3],[-1,-20,1.5]])

hadamard = np.multiply(A,B)
# print(hadamard)

AB_T = np.matmul(A, B.T)
# print(AB_T)

BA_T = np.matmul(B, A.T)
# print(BA_T)

x = np.array([1,0,1,0]).T
w = np.array([5,4,6,1]).T
w_t_x = np.matmul(w.T, x.T)
# print(w_t_x)
print(11**2)