
import numpy as np
import matplotlib.pyplot as plt

# Pseudorandom number generator using LCG (example from Table in Knuth's book)

# Parameters
m = 2**32
a = 1812433253
c = 1
n = 100000


''' Linear Congruential Generator (normalized to [0,1])
   
    n (int): is number of terms to generate
    
    Returns:
    list: array of pseudorandom integers'''


def LCG(a, m, c, n):
    
    X = np.zeros(n, dtype=np.uint64)
    for i in range(1, n):
        X[i] = (a * X[i-1] + c) % m
        
    return X/m


# Generate the random numbers
lcg= LCG(a, m, c, n)


''' Extract two sequences (weakly independent) from the LCG'''

def U1(a,m,c,n):
    n2=n//2
    X=np.zeros(n2)
    Y=LCG(a, m, c, n)
    for i in range(0,n2):
        X[i]=Y[2*i+1]

    return X

def U2(a,m,c,n):
    n2=n//2
    X=np.zeros(n2)
    Y=LCG(a, m, c, n)
    for i in range(0,n2):
        X[i]=Y[2*i]

    return X


lcgodd= U1(a, m, c, n)
lcgeven= U2(a, m, c, n)


'''Generate two weakly independent normally distr. rv
    notice that since start at zero and period m we never hit 0 again'''




def Z1(a,m,c,n):
    n2=int(n/2)
    Z=np.zeros(n2)
    U=U1(a, m, c, n)
    V=U2(a, m, c, n)
    for i in range(1,n2):
        Z[i]=np.sqrt(-2 * np.log(U[i])) * np.cos(2 * np.pi * V[i])

    return Z



def Z2(a,m,c,n):
    n2=int(n/2)
    Z=np.zeros(n2)
    U=U1(a, m, c, n)
    V=U2(a, m, c, n)
    for i in range(1,n2):
        Z[i]=np.sqrt(-2 * np.log(U[i])) * np.sin(2 * np.pi * V[i])

    return Z


Zlist1=Z1(a,m,c,n)

# Plot histogram of Z1
plt.figure(figsize=(10, 6))
plt.hist(Zlist1, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Z1 (Standard Normal Distribution)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()



