import numpy as np
import matplotlib.pyplot as plt





'''Price Bonds using the CIR model for evolution of interest rates'''

    #Approach1: via discretization
    #Approach2: exact simulation
    #Approach3: exact formula



'''APPROACH1'''

    #Euler discretization of Feller Square-root process for Short interest rates

    #takes as inputs r0 initial value, T time interval, parameters a and b, sigma volatility risky,
            #n time steps, N number of paths.
    #r is an internal variable which stands for the asset price.     

def EulerFellerSQRT(r0, T, a,b, sigma, n, N):
    dt = T / n
    r = np.zeros((n + 1, N))
    r[0] = r0
    for t in range(1, n + 1):
        r[t] = r[t-1]+a*(b-r[t])*dt +  sigma* np.sqrt(r[t-1]*dt) * np.random.standard_normal(N)
    return r #time array of interest rates




def MCdiscretized(r0, T, a,b, sigma, n, N):
    r= EulerFellerSQRT(r0, T, a,b, sigma, n, N) #Discretized process
    B=np.exp(-sum(r)*T/n)        #Bond prices individual paths following CIR model
    return np.mean(B),B        #B and Expectation approx.









'''APPROACH2'''

    #Use exact formula for transition densities (see Glasserman)


def transitionSQRT(r0,T,a,b,sigma,n,N):
    dt = T / n
    A=(sigma**2)*(1-np.exp(-a*dt)) / (4*a)
    B=4*a*np.exp(-a*dt) / ((sigma**2)*(1-np.exp(-a*dt)))    #non-centralityparameter (divided by r[t])
    d=4*b*a/sigma**2                                     #degrees of freedom for chi-squared distribution

    r=np.zeros((n+1,N))
    r[0]=r0
    
    for t in range(1,n+1):
        r[t]=A*np.random.noncentral_chisquare(d,B*r[t-1], N)

    return r #time array of interest rates



def MCtransition(r0, T, a,b, sigma, n, N):
    r= transitionSQRT(r0, T, a,b, sigma, n, N) #Discretized process
    B=np.exp(-sum(r)*T/n)        #Bond prices individual paths following CIR model
    return np.mean(B),B        #B and Expectation approx.





'''APPROACH 3'''
                         
def BPRICE(r0,T,a,b,sigma,n):
    h=np.sqrt(a**2+2*sigma**2)
    A=((2*h*np.exp((a+h)*T/2))/(2*h+(a+h)*(np.exp(h*T)-1)))**(2*a*b/sigma**2)
    B=2*(np.exp(h*T)-1)/(2*h+(a+h)*(np.exp(h*T)-1))
    return A*np.exp(-B*r0)
                    
                           

'''EXAMPLES'''





# PARAMETERS
r0 = 0.045
T = 1.0
a = 0.3
b= 0.05
sigma = 0.08
n = 365
N = 300000


MCBond,B_discr=MCdiscretized(r0, T, a,b, sigma, n, N)
MCBond2,B_trans=MCtransition(r0, T, a,b, sigma, n, N)
MCBond3=BPRICE(r0, T, a,b, sigma, n)

print('Price Bond with approach 1', MCBond)
print('Price Bond with approach 2', MCBond2)
print('Exact Price Bond', MCBond3)



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(B_discr, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Bond Prices (Discretized CIR)")
plt.xlabel("Bond Price")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(B_trans, bins=50, color='salmon', edgecolor='black', alpha=0.7)
plt.title("Histogram of Bond Prices (Exact CIR Transition)")
plt.xlabel("Bond Price")

plt.tight_layout()
plt.show()



