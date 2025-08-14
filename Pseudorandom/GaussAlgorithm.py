import numpy as np

def gauss_algorithm(v1, v2):
    u = [np.array(v1, dtype=float), np.array(v2, dtype=float)]
    
    while True: #runs until we hit break
        # Sort vectors by length (ensure ||u0|| <= ||u1||)
        if np.linalg.norm(u[0]) > np.linalg.norm(u[1]):
            u[0], u[1] = u[1], u[0]
        
        # Compute projection coefficient
        m = round(np.dot(u[0], u[1]) / np.dot(u[0], u[0]))
        
        # Termination condition: m == 0 (m has ben rounded, before rounding m\leq 1/2)
        if m == 0:
            break
        
        # Reduce u[1] by m*u[0]
        u[1] = u[1] - m * u[0]
    
    return u

# Test from Hoffstein
v1 = [66586820, 65354729]
v2 = [6513996, 6393464]
result = gauss_algorithm(v1, v2)
print('First example:', (v1,v2))
print('reduced basis:', result)

# Test from Knuth
v1 = [10**10,0]
v2 = [-3141592621,1]
result = gauss_algorithm(v1, v2)
print('Second example:', (v1,v2))
print('reduced basis:', result)
