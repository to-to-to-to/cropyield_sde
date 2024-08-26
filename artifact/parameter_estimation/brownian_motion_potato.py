import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Generate Brownian motion for crop yield
W = BrownianMotion(seed, N)

# Plot the Brownian motion
plt.figure(figsize=(16, 8))
plt.plot(years, W, color="green")
plt.title('Discretized 1D Brownian Motion for Crop Yielding (Potato)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Brownian increment $W(t)$', fontsize=12)
axes = plt.gca()
axes.set_xlim([years[0], years[-1]])  # Set x-axis limits to match the range of years
plt.show()