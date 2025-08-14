import numpy as np




'''LLL ALGORITHM'''




   #Gram Schmidt subroutine

def gram_schmidt(V):
    n = len(V)
    U = [np.array(v, dtype=float) for v in V]  # Copy input vectors
    M = np.zeros((n, n))  # Matrix for projection coefficients
    
    for j in range(n):
        for i in range(j):
            M[i, j] = np.dot(U[i], U[j]) / np.dot(U[i], U[i])
            U[j] = U[j] - M[i, j] * U[i]
    
    return U, M



    #LLL algorithm
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






'''COMPUTATION OF DUAL LATTICE'''




def lattice_basis(a, m, n):
    V = np.zeros((n, n))
    for i in range(n):
        V[i,0] =  a ** i/ m # First column: (1/m, a/m, a²/m, ...)
    for i in range(1,n):
        V[i,i]=1     # Identity submatrix
    return V

def dual_lattice(a, m, n):
    """Computes the dual lattice basis"""
    V = lattice_basis(a, m, n).T #Transpose
    return np.linalg.inv(V)      #Inverse of transpose




'''SHORTEST VECTOR AND SPECTRAL TEST'''


def shortest_vector(a, m, n):
  
    basis = [dual_lattice(a,m,n).T[i] for i in range(n)]  #We need this because lll-algor works with lists of vectors not with matrices
    reduced_basis = lll_algorithm(basis)
    return reduced_basis[0]  # First vector in LLL-reduced basis

def spectral_test(a, m, n):
 
    v = shortest_vector(a, m, n)
    return 1 / np.linalg.norm(v)  # 1 / ||v||



'''EXAMPLES '''
    #Knuth

a = 3141592621
m = 10**10

SV=shortest_vector(a,m,3)
print('SV=', SV)

    #verify that this vector belongs to the lattice
z=  np.linalg.solve( lattice_basis(a,m,3), SV)

print(z)
z_round=np.round(z)
Error=z_round-z
if np.allclose(Error,0):
    print('SV belongs to L')

else: print('SV does not belong to L')



