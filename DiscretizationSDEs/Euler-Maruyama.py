import numpy as np
import matplotlib.pyplot as plt



'''Euler discretization of GBM'''

    #takes as inputs S0 initial price, T time interval, r interest rate riskless, sigma volatility risky,
            #n time steps, N number of paths.
    #S is an internal variable which stands for the asset price. We evolve it under the RISK-NEUTRAL dynamics.
    

def EulerGBM(S0, T, r, sigma, n, N):
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0
    for t in range(1, n + 1):
        S[t] = S[t-1]*(1 + (r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(N))
    return S #Final price

# PARAMETERS
S0 = 100.0
T = 1.0
r = 0.05
sigma = 0.2
n = 365
N = 300000

# SIMULATION OF N PATHS OF GBM
S= EulerGBM(S0, T, r, sigma, n, N)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(S[:, :5])  # Plot first 5 paths
plt.title('GBM Sample Paths')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.grid(True)
plt.show()



