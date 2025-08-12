import numpy as np
import matplotlib.pyplot as plt



#PRICE EUROPEAN CALL OPTION USING MC, GIVE CONFIDENCE INTERVAL.

    #Euler Discretization
    #Pricing by MonteCarlo




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
N = 3000000

# SIMULATION OF N PATHS OF GBM
S= EulerGBM(S0, T, r, sigma, n, N)




#MONTE CARLO PRICING OF EUROPEAN OPTION

K=100   #Strike price 

    #We now compute the discounted payoff(for each path).
    #We do it in a vectorized way i.e. D_T is an array

D_T=np.exp(-r*T)*np.maximum(S[n]-K,0)


D=D_T.mean() #Mean of discounted payoffs (Monte Carlo approximation to option price)
print('Price European option:', D)

#Confidence interval of 95%
conf_interval = D + np.array([-1.96, 1.96]) * D_T.std() / np.sqrt(N)

print('Confidence Interval:', conf_interval)
