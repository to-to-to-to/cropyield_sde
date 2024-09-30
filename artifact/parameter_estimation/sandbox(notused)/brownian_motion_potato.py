import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heapq

# Load the dataset
file_path = '../datasets/CropSDEData/YIELD_NUTS0_NL.csv'
data = pd.read_csv(file_path)

# Filter the data for the crop "potato"
potato_data = data[data['CROP'] == 'potato']

# Extract the yield data and the corresponding years
yield_data = potato_data['YIELD'].values
years = potato_data['FYEAR'].values

# Parameters for the Wiener process in the crop yielding case
T = len(years)  # Time interval corresponds to the number of years in the dataset
N = T  # Number of points, one per year

seed = 20  # Seed for reproducibility

dt = T / N  # Time step
t = np.linspace(years[0], years[-1], N)  # Time vector from start year to end year

# Function to generate Brownian motion
def BrownianMotion(seed, N):
    np.random.seed(seed)
    Z = np.random.randn(N)  # Random variables
    Z[0] = 0  # Start at 0
    dW = np.sqrt(dt) * Z  # Single Brownian increment
    W = np.cumsum(dW)  # Brownian path (cumulative sum)
    return W

# Function to generate Euler-Maruyama increments
def EulerMaruyamaIncrements(W, mu, sigma, dt):
    N = len(W)
    X = np.zeros(N)
    X[0] = W[0]
    
    for i in range(1, N):
        dW = W[i] - W[i-1]  # Increment in Brownian motion
        dX = mu * dt + sigma * dW
        X[i] = X[i-1] + dX
    
    return X

# Generate Brownian motion for crop yield
W = BrownianMotion(seed, N)

# Define a broader range of values for mu and sigma with smaller increments
mu_values = np.linspace(0.0001, 0.001, 100)  # 100 values between 0.0001 and 0.001
sigma_values = np.linspace(0.05, 0.15, 100)  # 100 values between 0.05 and 0.15

# List to store the results
results = []

# Loop over all combinations of mu and sigma
for mu in mu_values:
    for sigma in sigma_values:
        X_em = EulerMaruyamaIncrements(W, mu, sigma, dt)
        error = np.sum(np.abs(X_em - W))  # Calculate the error
        heapq.heappush(results, (error, mu, sigma, X_em))  # Store the error, parameters, and result

# Get the best approximation
best_approximation = heapq.nsmallest(1, results)[0]

# Extract the best parameters and the approximation
best_error, best_mu, best_sigma, best_X_em = best_approximation

# Plot the Brownian motion
plt.figure(figsize=(16, 8))
plt.plot(years, W, color="green", label="Brownian Motion")

# Plot the best approximation
plt.plot(years, best_X_em, color="blue", linestyle="--", label=f'Best EM Approximation (mu={best_mu:.4f}, sigma={best_sigma:.3f})')

plt.title('Brownian Motion and Best Euler-Maruyama Approximation for Crop Yielding (Potato)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Brownian increment $W(t)$', fontsize=12)
plt.legend(fontsize=12)
axes = plt.gca()
axes.set_xlim([years[0], years[-1]])  # Set x-axis limits to match the range of years
plt.show()
