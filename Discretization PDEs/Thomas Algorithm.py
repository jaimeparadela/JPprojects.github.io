
import numpy as np

def thomas_algorithm(a, b, c, d):
    n = len(d)
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()
    
    # Forward elimination
    for i in range(1, n):
        if b[i-1] == 0:
            raise ValueError("Division by zero")
        m = a[i-1] / b[i-1]
        b[i] =b[i]- m * c[i-1]
        d[i] = d[i]- m * d[i-1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

# Original data 
a_orig = np.array([-1, -1, -1], dtype=float)  #dtype=float otherwise if integer entries operations with these arrays will cut decimals#
b_orig = np.array([ 2,  3,  2,  2], dtype=float)
c_orig = np.array([-1, -1, -1], dtype=float)
d_orig = np.array([ 1,  0,  6,  1], dtype=float)

# Solve
x = thomas_algorithm(a_orig, b_orig, c_orig, d_orig)
print("Solution:", x)

# Build A for verification
n = len(d_orig)
A = np.zeros((n, n))
for i in range(n):
    if i > 0:
        A[i, i-1] = a_orig[i-1]
    A[i, i] = b_orig[i]
    if i < n-1:
        A[i, i+1] = c_orig[i]

Error = A @ x - d_orig
print("Error:", Error)

