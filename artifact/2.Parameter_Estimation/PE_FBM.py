import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma

# Load dataset
df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")
df = df.dropna(subset=['TAVG'])

# Compute log-returns of TAVG (temperature)
df['log_TAVG'] = np.log(df['TAVG'])
df['returns'] = df['log_TAVG'].diff().dropna()

# Estimate rolling variance
rolling_window = 30  # 30 periods to smooth the variance
df['variance'] = df['returns'].rolling(window=rolling_window).var().dropna()

# Drop NaN values
df = df.dropna(subset=['returns', 'variance'])

# Define the fBM log-likelihood function
def fbm_log_likelihood(params, returns):
    """
    Compute the log-likelihood for the Fractional Brownian Motion (fBM) model.
    """
    mu, sigma, H = params
    log_likelihood = 0
    T = len(returns)
    
    # Compute the covariance matrix of fBM
    cov_matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            cov_matrix[i, j] = 0.5 * (abs(i+1)**(2*H) + abs(j+1)**(2*H) - abs(i-j)**(2*H))

    # Compute the log-likelihood assuming multivariate normal distribution
    cov_matrix *= sigma ** 2
    det_cov = np.linalg.det(cov_matrix + np.eye(T) * 1e-6)  # Add small value for numerical stability
    inv_cov = np.linalg.inv(cov_matrix + np.eye(T) * 1e-6)

    # Compute likelihood
    returns_centered = returns - mu
    log_likelihood = -0.5 * np.log(det_cov) - 0.5 * np.dot(returns_centered.T, np.dot(inv_cov, returns_centered))
    
    return -log_likelihood  # Negative for minimization

# Initial parameter guesses
initial_params = [0.01, 0.2, 0.5]  # mu, sigma, H

# Define bounds for parameters to ensure they remain physically meaningful
param_bounds = [(None, None),  # mu (drift) can be any value
                (1e-4, None),  # sigma (volatility) must be positive
                (0.01, 0.99)]  # Hurst parameter H (0 < H < 1)

# Perform MLE optimization
result_constrained = minimize(fbm_log_likelihood, initial_params, args=(df['returns'],), 
                              method='L-BFGS-B', bounds=param_bounds)

# Extract estimated parameters
mu_hat, sigma_hat, H_hat = result_constrained.x

# Display estimated parameters
estimated_params_constrained = {
    "Mu (Drift)": mu_hat,
    "Sigma (Volatility)": sigma_hat,
    "Hurst Parameter (H)": H_hat
}

print("Estimated fBM Parameters for TAVG:")
print(estimated_params_constrained)
