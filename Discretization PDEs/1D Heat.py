import numpy as np

import matplotlib.pyplot as plt


#Solve u_t=nu u_xx on (x,t)\in... and with:

    #IC u(0,x)= sin(2pix)
    #BC u(t,0)=0 u(t,1)=0



# Represent numerical solution as N+1\times K+1 matrix


def u(N, K, nu, deltaT):
    u=np.zeros((N+1,K+1)) #Initialize the matrix

    deltaX=1/K

    kappa=nu*deltaT/(deltaX**2) #Adimensional parameter
    
    for k in range(0, K+1):
        u[0,k]=np.sin(2*np.pi*k*deltaX) #Initial condition

    for n in range (1,N+1):
        for k in range (1,K):
            u[n,k]=u[n-1,k]+kappa*(u[n-1,k+1]-2*u[n-1,k]+u[n-1,k-1])

        
    return u





#Analytical Solution

def v(N, K, nu, deltaT):
    v=np.zeros((N+1,K+1)) #Initialize the matrix

    deltaX=1/K


    for n in range (0,N+1):
        for k in range (0,K+1):
            v[n,k]=np.exp(-4*(np.pi**2)*nu*n*deltaT)*np.sin(2*np.pi*k*deltaX)
        
    return v








#PARAMETERS (random choice, below we present a better choice)

N=10000
K=10
deltaT=0.02
nu=1/6

U=u(N,K,nu,deltaT)




V=v(N,K,nu,deltaT)



#VISUALIZE DATA


# Plot solution profile at specific times

plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, K+1)  # Spatial grid

# Plot certain time profile



m= 40 # Choose 0\leq m\leq N



plt.plot(x, V[m, :], color='blue')



plt.plot(x, U[m,:], color='red')

plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Parabolic Solution")
plt.grid(True)
plt.show()




#PARAMETERS (now we choose a better relation between deltaT, deltaX and nu)



Kopt=int(np.floor(1/np.sqrt(6*nu*deltaT)))       #OPTIMAL DELTAX: This ensures that deltaT=deltaX**2/(6nu) 

                                      

U=u(N,Kopt,nu,deltaT)  #SOLUTION OPTIMAL DELTAX DELTAT

W=u(N,Kopt,nu,deltaT/100) #SMALLER DELTAT BUT NOT OPTIMAL CHOICE


V=v(N,Kopt,nu,deltaT)



#VISUALIZE DATA


# Plot solution profile at specific times

plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, Kopt+1)  # Spatial grid

# Plot certain time profile



m= 40 # Choose 0\leq m\leq N



plt.plot(x, U[m, :], color='blue')

plt.plot(x, W[100*m, :], color='green') #Since DeltaT is smaller we need to evaluate at later time


plt.plot(x, V[m,:], color='red')

plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Parabolic Solution")
plt.grid(True)
plt.show()






