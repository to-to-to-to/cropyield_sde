import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load dataset
df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")

df = df.dropna(subset=['TAVG'])

# Compute log-returns of TAVG (temperature)
df['log_TAVG'] = np.log(df['TAVG'])
df['returns'] = df['log_TAVG'].diff().dropna()

# Estimate rolling variance (stochastic variance process V_t)
rolling_window = 30  # 30 periods to smooth the variance
df['variance'] = df['returns'].rolling(window=rolling_window).var().dropna()

# Drop NaN values
df = df.dropna(subset=['returns', 'variance'])

# Define the Heston log-likelihood function
def heston_log_likelihood(params, returns, variance):
    """
    Compute the log-likelihood for the Heston Model.
    """
    mu, kappa, theta, sigma = params
    log_likelihood = 0
    T = len(returns)
    
    for t in range(1, T):
        vt = variance.iloc[t]
        vt_minus_1 = variance.iloc[t - 1]

        # Expected variance at time t
        expected_vt = vt_minus_1 + kappa * (theta - vt_minus_1)
        variance_vt = sigma**2 * vt_minus_1

        # Compute log-likelihood
        if variance_vt > 0 and vt > 0:
            ll_v = -0.5 * np.log(2 * np.pi * variance_vt) - ((vt - expected_vt) ** 2) / (2 * variance_vt)
            ll_r = -0.5 * np.log(2 * np.pi * vt) - (returns.iloc[t] - mu) ** 2 / (2 * vt)
            log_likelihood += ll_v + ll_r

    return -log_likelihood  # Negative for minimization

# Initial parameter guesses
initial_params = [0.01, 0.2, 0.02, 0.1]

# Define bounds for parameters to ensure they remain physically meaningful
param_bounds = [(None, None),  # mu (drift) can be any value
                (1e-4, None),  # kappa (mean reversion speed) must be positive
                (1e-4, None),  # theta (long-run variance) must be positive
                (1e-4, None)]  # sigma (volatility of volatility) must be positive

# Re-run optimization with bounds
result_constrained = minimize(heston_log_likelihood, initial_params, args=(df['returns'], df['variance']), 
                              method='L-BFGS-B', bounds=param_bounds)

# Extract new estimated parameters
mu_hat, kappa_hat, theta_hat, sigma_hat = result_constrained.x

# Display updated estimated parameters
estimated_params_constrained = {
    "Mu (Drift)": mu_hat,
    "Kappa (Mean Reversion Speed)": kappa_hat,
    "Theta (Long-run Variance)": theta_hat,
    "Sigma (Volatility of Volatility)": sigma_hat
}

print(estimated_params_constrained)
