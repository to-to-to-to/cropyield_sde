import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '../datasets/CropSDEData/YIELD_NUTS0_NL.csv'  # Ensure the file path is correct
data = pd.read_csv(file_path)

# Filter the data for the crop "potato"
potato_data = data[data['CROP'] == 'potato']

# Extract the yield data and the corresponding years
yield_data = potato_data['YIELD'].values
years = potato_data['FYEAR'].values

# Function to calculate yearly returns
def yearly_return(yield_data):
    returns = []
    for i in range(len(yield_data) - 1):
        this_year = yield_data[i + 1]
        last_year = yield_data[i]
        annual_return = (this_year - last_year) / last_year
        returns.append(annual_return)
    return returns

# Function to apply GBM for a specific period
def GBM_period(So, mu, sigma, seed, N):
    t = np.linspace(0., N-1, N)
    W = BrownianMotion(seed, N)
    
    S = [So]
    for i in range(1, N):
        drift = mu * t[i]  # Deterministic part
        diffusion = sigma * W[i-1]  # Stochastic part
        S_i = S[-1] * np.exp(drift + diffusion)  # GBM formula
        S.append(S_i)
    return np.array(S)

# Generate Brownian motion (Wiener process)
def BrownianMotion(seed, N):
    np.random.seed(seed)
    return np.random.normal(0, 1, N-1)  # N-1 because we're generating increments

# Calculate the returns, mean (mu) and volatility (sigma)
returns = yearly_return(yield_data)
mu = np.mean(returns)
sigma = np.std(returns)

# Splitting the data into intervals (e.g., 4-year periods)
interval = 4
num_periods = len(yield_data) // interval

# Try different seeds in the range of 1 to 100
seeds = range(1, 101)
best_seed = None
best_rmse = float('inf')
best_simulated_yields = None
simulation_results = []

for seed in seeds:
    simulated_yields = []
    
    for i in range(num_periods):
        start_idx = i * interval
        end_idx = start_idx + interval
        if end_idx > len(yield_data):
            end_idx = len(yield_data)
        
        So = yield_data[start_idx]
        period_yield_data = yield_data[start_idx:end_idx]
        
        simulated_period = GBM_period(So, mu, sigma, seed=seed, N=len(period_yield_data))
        simulated_yields.extend(simulated_period)
    
    simulated_yields = np.array(simulated_yields)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((simulated_yields - yield_data[:len(simulated_yields)]) ** 2))
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_seed = seed
        best_simulated_yields = simulated_yields
    
    # Store the result for further analysis
    simulation_results.append((seed, rmse, simulated_yields))

# Plot the best simulation
plt.figure(figsize=(16, 8))
plt.plot(years, yield_data, label='Actual Yield', color='blue')
plt.plot(years[:len(best_simulated_yields)], best_simulated_yields, label=f'Best Simulated Yield (GBM, Seed={best_seed})', color='orange', linestyle='--')
plt.title('Comparison of Actual Yield with Best GBM Simulation', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Yield (Tonnes per hectare)', fontsize=12)
plt.legend(loc='upper left')
plt.show()

# Now sort and plot the top 5 simulations
top_5_simulations = sorted(simulation_results, key=lambda x: x[1])[:5]

# Initialize plot for top 5 seeds
plt.figure(figsize=(16, 8))

# Plot actual yield
plt.plot(years, yield_data, label='Actual Yield', color='blue')

# Plot top 5 simulations
for seed, rmse, simulated_yields in top_5_simulations:
    plt.plot(years[:len(simulated_yields)], simulated_yields, linestyle='--', label=f'Seed={seed}, RMSE={rmse:.2f}')

# Final plot adjustments
plt.title('Comparison of Actual Yield with Top 5 GBM Simulations', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Yield (Tonnes per hectare)', fontsize=12)
plt.legend(loc='upper left')
plt.show()

# Plot comparison with seeds 1 to 50
plt.figure(figsize=(16, 8))

# Plot actual yield
plt.plot(years, yield_data, label='Actual', color='black', linewidth=2)

# Plot other simulations (seeds 1 to 50)
for seed in range(1, 101):
    simulated_yields = []
    
    for i in range(num_periods):
        start_idx = i * interval
        end_idx = start_idx + interval
        if end_idx > len(yield_data):
            end_idx = len(yield_data)
        
        So = yield_data[start_idx]
        period_yield_data = yield_data[start_idx:end_idx]
        
        simulated_period = GBM_period(So, mu, sigma, seed=seed, N=len(period_yield_data))
        simulated_yields.extend(simulated_period)
    
    simulated_yields = np.array(simulated_yields)
    
    plt.plot(years[:len(simulated_yields)], simulated_yields, linestyle='--')

# Final plot adjustments
plt.title('Comparison of Actual Yield with Multiple GBM Simulations (Seeds 1-100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Yield (Tonnes per hectare)', fontsize=12)
plt.legend(['Actual'], loc='upper left')  # Only show 'Actual' in the legend
plt.show()

# Euler-Maruyama Approximation for Crop Yield
def Em(So, mu, sigma, W, T, TS):
    dt = T / TS  # EM step size (T divided by the number of steps)
    wi = [So]
    for i in range(1, min(TS, len(W))):  # Ensure loop doesn't exceed length of W
        delta_Wi = W[i] - W[i - 1]
        wi_new = wi[-1] + mu * wi[-1] * dt + sigma * wi[-1] * delta_Wi
        wi.append(wi_new)
    return np.array(wi), dt

# Function to calculate the sum of squared differences (SSD) between GBM and EM
def calculate_ssd(actual, predicted):
    min_len = min(len(actual), len(predicted))
    return np.sum((actual[:min_len] - predicted[:min_len]) ** 2)

# Simulation parameters
N = len(years)  # Number of intervals (equal to number of years)
So = yield_data[0]  # Initial yield (first value in the dataset)

# Adjusting mu and sigma for better trend capture
mu = np.mean(np.diff(yield_data) / yield_data[:-1])  # Recalculate mu based on your data
sigma = np.std(np.diff(yield_data) / yield_data[:-1])  # Recalculate sigma based on your data

# List to store the top 3 results
top_results = []

# Loop through seeds from 1 to 100
for seed in range(1, 101):
    np.random.seed(seed)
    W = np.cumsum(np.random.normal(0, 1, N - 1))  # Generate Brownian motion
    
    # Loop through different time steps (TS) to find the best one
    for TS in range(50, 201, 50):  # Adjust the range and step size as needed
        # Perform Euler-Maruyama approximation
        Approx, dt = Em(So, mu, sigma, W, T=N, TS=TS)  # Using N (number of years) as T
        
        # Calculate SSD to evaluate the performance
        ssd = calculate_ssd(best_simulated_yields, Approx)
        
        # Store the result
        top_results.append((ssd, seed, TS, dt, Approx))

# Sort the results by SSD to find the top 3
top_results.sort(key=lambda x: x[0])
top_3 = top_results[:3]

# Plotting the results using the top 3 seeds
plt.figure(figsize=(16, 8))

# Plot the best GBM simulation from your previous code
plt.plot(years[:len(best_simulated_yields)], best_simulated_yields, label=f'Best Simulated Yield (GBM)', color='purple', linewidth=2)

# Plot the top 3 Euler-Maruyama approximations
colors = ['orange', 'green', 'blue']
for i, (ssd, seed, TS, dt, Approx) in enumerate(top_3):
    plt.plot(years[:len(Approx)], Approx, label=f'Top {i+1} Euler-Maruyama (Seed={seed}, TS={TS}, dt={dt:.3f}, SSD={ssd:.2f})', color=colors[i], linestyle='--')

# Labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Crop Yield (Tonnes per hectare)', fontsize=12)
plt.title('Top 3 Euler-Maruyama Approximations Compared to GBM', fontsize=14)

# Show legend and plot
plt.legend(loc='upper left')
plt.show()

# Output the top 3 details
for i, (ssd, seed, TS, dt, _) in enumerate(top_3):
    print(f"Top {i+1} Euler-Maruyama Approximation: Seed={seed}, TS={TS}, dt={dt:.3f}, SSD={ssd:.2f}")