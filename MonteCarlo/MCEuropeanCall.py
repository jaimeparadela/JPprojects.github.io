import numpy as np
import matplotlib.pyplot as plt



#PRICE EUROPEAN CALL OPTION USING MC, GIVE CONFIDENCE INTERVAL.

    #Euler Discretization
    #Pricing by MonteCarlo: Confidence interval.
    #Computation of the delta by MonteCarlo: 2 approaches (valid in BS model)
        # Direct definition formula (differentiable payoff, except count set)
        # A la Malliavin formula
        
    # Observe variance reduction as T to 0 (see Pagès).




'''Euler discretization of GBM'''

    #takes as inputs S0 initial price, T time interval, r interest rate riskless, sigma volatility risky,
            #n time steps, N number of paths.
    #S is an internal variable which stands for the asset price. We evolve it under the RISK-NEUTRAL dynamics.
    

def EulerGBM(S0, T, r, sigma, n, N):  #In the code we also compute the evolution of the BM process along same path (needed for second method for computing delta)
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    B=np.zeros((n+1,N))
    B[0]=0
    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1]*(1 + (r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) *deltaB)
        B[t]= B[t-1]+np.sqrt(dt)*deltaB

        
    return S,B #Price and BM arrays









#MONTE CARLO PRICING OF EUROPEAN OPTION PLUS DELTA


def MCEuropean_price_and_delta(K,S0, T, r, sigma, n, N):
    S = np.zeros((n + 1, N))
    B=np.zeros((n+1,N))
    S,B=EulerGBM(S0, T, r, sigma, n, N)

    ##OPTIONPRICE:
    C_T=np.exp(-r*T)*np.maximum(S[n]-K,0)
    C=C_T.mean() #Mean of discounted payoffs (Monte Carlo approximation to option price)
    conf_interval = C + np.array([-1.96, 1.96]) * C_T.std() / np.sqrt(N) #95%confidence interval

    

    ##APPROACH 1 computation of delta: DELTA=np.exp(-r*T)*E(\partial_{S_0}(varphi(S_T)) *S_T/S_0) (FORMULA VALID IN BS MODEL, DIFF PAYOFF EXCEPT COUNT SET)

    def step_K(x,K):
        return np.where(x>K,1,0) #works in vectorized way: if x[i]>K, returns one in that position, otherwise returns 0 in that position
    
    delta1path=np.exp(-r*T)*step_K(S[n],K)*S[n]/S0 #pathwisevalues
    delta1=delta1path.mean()
    conf_interval_delta1 = delta1 + np.array([-1.96, 1.96]) * delta1path.std() / np.sqrt(N)

    #APPROACH 2: DELTa=np.exp(-r*T)*E(\varphi(S_T) *B_T/(S0*sigma*T)) (FORMULA VALID IN BS MODEL, BOREL PAYOFF)
    delta2path=np.exp(-r*T)*np.maximum(S[n]-K,0)*B[n]/(S0*sigma*T)
    delta2=delta2path.mean()
    conf_interval_delta2 = delta2 + np.array([-1.96, 1.96]) * delta2path.std() / np.sqrt(N)

    return C,conf_interval,delta1,conf_interval_delta1,delta2,conf_interval_delta1,delta1path,delta2path #delta1path and delta2path for study variancereduction




# PARAMETERS and SIMULATION
S0 = 100.0
T = 100
r = 0.05
sigma = 0.2
n = 300
N = 300


K=100   #Strike price 



C,conf_interval,delta1,conf_interval_delta1,delta2,conf_interval_delta2,delta1path,delta2path=MCEuropean_price_and_delta(K,S0, T, r, sigma, n, N)



print('Price European option:', C)

print('Confidence Interval:', conf_interval)

print('Delta for European option by Approach1',delta1)

print('Confidence Interval for Delta1:', conf_interval_delta1)

print('Delta for European option by Approach2',delta2)

print('Confidence Interval for Delta2:', conf_interval_delta2)



# New parameters (from Pages): Variance as T to 0 for both computations of delta

S0 = 100.0
r = 0
sigma = 0.5
n = 300
N = 300
M=100 #discretization of the interval T in [0,1]

K=95   #Strike price

def VarrespecttoT(K,S0,r,sigma,n,N,M):
    delta1var=np.zeros((M,N))
    delta2var=np.zeros((M,N))
    Tlist=np.zeros(M)
    for t in range(1,M+1):

        Tlist[t-1]=0.1*t/M

        delta1path=MCEuropean_price_and_delta(K,S0, Tlist[t-1], r, sigma, n, N)[6] #retrieve delta1
        delta2path=MCEuropean_price_and_delta(K,S0, Tlist[t-1], r, sigma, n, N)[7] #retrieve delta2

        delta1var[t-1]=delta1path.std()**2
        delta2var[t-1]=delta2path.std()**2

    return delta1var, delta2var,Tlist



delta1var,delta2var,Tlist=VarrespecttoT(K,S0,r,sigma,n,N,M)



# Visualization
plt.figure(figsize=(10, 6))


plt.plot(Tlist, delta2var, 's-', linewidth=2, markersize=5, label='Malliavin weight', color='blue')  
plt.plot(Tlist, delta1var, 'o-', linewidth=2, markersize=5, label='Definition', color='red')
plt.xlabel('T')
plt.ylabel('Variance')
plt.title('Variance Reduction')
plt.grid(True, alpha=0.3)
plt.show()
    
    

    
