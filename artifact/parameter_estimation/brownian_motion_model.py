import numpy as np
import matplotlib.pyplot as plt
import sdeint

# Set parameters for the Geometric Brownian Motion

'''
- mu: Drift coefficient (positive trend)
- sigma: Volatility (adjusted for smoother fluctuations)
- T: Time period
- dt: Time step
- N: #of steps
- t: Time Vector
- X0: initial cropyield
'''

mu = 0.05  
sigma = 0.03  
T = 10 
dt = 0.01 
N = int(T/dt)
t = np.linspace(0, T, N)
X0 = 10

# Define the SDE: dX = mu * X * dt + sigma * X * dW
def f(X, t):
    return mu * X

def g(X, t):
    return sigma * X

# Simulate the process
X = sdeint.itoint(f, g, X0, t)

# Manually create the shading area at the end -> represents uncertainty of possible outcomes
shade_start_index = int(0.9 * N)
shade_x = t[shade_start_index:]
shade_y = X[shade_start_index:]
shade_y = shade_y.flatten()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, X, label='Stochastic Crop Yield', color='red')
plt.fill_between(shade_x, 0, shade_y, color="red", alpha=0.4)
plt.title('GBM for Crop Yielding')
plt.xlabel('Time (Years)')
plt.ylabel('Crop Yield')
plt.ylim([0, max(X)*1.1]
plt.xlim([0, T])
plt.show()
