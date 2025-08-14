import numpy as np


'''Gram Schmidt subroutine'''

def gram_schmidt(V):
    n = len(V)
    U = [np.array(v, dtype=float) for v in V]  # Copy input vectors
    M = np.zeros((n, n))  # Matrix for projection coefficients
    
    for j in range(n):
        for i in range(j):
            M[i, j] = np.dot(U[i], U[j]) / np.dot(U[i], U[i])
            U[j] = U[j] - M[i, j] * U[i]
    
    return U, M


V=[(1,1,0),(1,0,1),(0,0,1)]
[U,M]=gram_schmidt(V)
print('Original basis:',V,'Orthogonal basis:',U)


'''LLL algorithm'''
def lll_algorithm(V, delta=0.75):
    n = len(V)
    U = [np.array(v, dtype=float) for v in V]  # Copy input vectors
    W, M = gram_schmidt(U)  # Initial Gram-Schmidt
    
    k = 1  
    
    while k < n:
        # Size reduction
        for j in range(k-1, -1, -1):
            if abs(M[j, k]) > 0.5:
                m = round(M[j, k])
                U[k] = U[k] - m * U[j]
                W, M = gram_schmidt(U)  # Recompute after reduction
        
        # Lovász condition
        norm_Wk = np.linalg.norm(W[k])**2
        norm_Wk_minus_1 = np.linalg.norm(W[k-1])**2
        if norm_Wk >= (delta - M[k-1, k]**2) * norm_Wk_minus_1:
            k =k+ 1  # Move to next vector
        else:
            # Swap vectors and recompute
            U[k], U[k-1] = U[k-1], U[k]
            W, M = gram_schmidt(U)
            k = max(1, k-1)  # Ensure k doesn't go below 1
    
    return U


#Test from book

V = [(19, 2, 32, 46, 3, 33), (15, 42, 11, 0, 3, 24), (43, 15, 0, 24, 
    4, 16), (20, 44, 44, 0, 18, 15), (0, 48, 35, 16, 31, 31), (48, 33,
     32, 9, 1, 29)];

U=lll_algorithm(V)
print(U)
