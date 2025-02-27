import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma

# Load dataset
df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")
df = df.dropna(subset=['RELH'])

# Compute log-returns of RELH (Humidity)
df['log_RELH'] = np.log(df['RELH'])
df['returns'] = df['log_RELH'].diff().dropna()

# Drop NaN values
df = df.dropna(subset=['returns'])

# Define the Variance Gamma log-likelihood function
def vg_log_likelihood(params, returns):
    """
    Compute the log-likelihood for the Variance Gamma (VG) Process.
    """
    mu, sigma, nu = params
    log_likelihood = 0
    T = len(returns)
    
    for t in range(1, T):
        delta_t = returns.iloc[t] - mu  # Drift adjusted return
        gamma_variance = np.random.gamma(shape=nu, scale=1.0)  # Gamma process sample
        variance_t = sigma * np.sqrt(gamma_variance)  # VG variance component
        
        # Compute log-likelihood
        if variance_t > 0:
            ll = -0.5 * np.log(2 * np.pi * variance_t**2) - (delta_t**2) / (2 * variance_t**2)
            log_likelihood += ll

    return -log_likelihood  # Negative for minimization

# Initial parameter guesses
initial_params = [0.0, 0.1, 0.1]  # mu, sigma, nu

# Define bounds for parameters to ensure they remain physically meaningful
param_bounds = [(None, None),  # mu (drift) can be any value
                (1e-4, None),  # sigma (volatility) must be positive
                (1e-4, None)]  # nu (VG variance parameter) must be positive

# Perform MLE optimization
result_constrained = minimize(vg_log_likelihood, initial_params, args=(df['returns'],), 
                              method='L-BFGS-B', bounds=param_bounds)

# Extract estimated parameters
mu_hat, sigma_hat, nu_hat = result_constrained.x

# Display estimated parameters
estimated_params_constrained = {
    "Mu (Drift)": mu_hat,
    "Sigma (Volatility)": sigma_hat,
    "Nu (VG Variance Parameter)": nu_hat
}

print("Estimated Variance Gamma Parameters for RELH:")
print(estimated_params_constrained)
