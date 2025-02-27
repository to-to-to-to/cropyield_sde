import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson

# Load dataset
df = pd.read_csv("/Users/vitorihaldijiran/Desktop/cropyield_sde/artifact/datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv")

# Drop missing values for PREC
df = df.dropna(subset=['PREC'])

# Compute log-returns of PREC
df['log_PREC'] = np.log(df['PREC'] + 1)  # Adding 1 to avoid log(0) issues
df['returns'] = df['log_PREC'].diff().dropna()

# Define the Merton Jump-Diffusion log-likelihood function
def merton_log_likelihood(params, returns):
    """
    Compute the log-likelihood for the Merton Jump-Diffusion Model.
    """
    mu, sigma, lambda_, delta = params
    log_likelihood = 0
    T = len(returns)
    
    for t in range(T):
        r_t = returns.iloc[t]

        # Mixture of normal distributions: Jump-Diffusion probability
        normal_component = norm.pdf(r_t, loc=mu, scale=sigma)
        jump_component = poisson.pmf(1, lambda_) * norm.pdf(r_t, loc=mu + delta, scale=sigma)

        # Total likelihood is a mix of jump and normal components
        likelihood = (1 - poisson.pmf(1, lambda_)) * normal_component + jump_component
        
        if likelihood > 0:
            log_likelihood += np.log(likelihood)

    return -log_likelihood  # Negative for minimization

# Initial parameter guesses
initial_params = [0.01, 0.2, 0.1, 0.05]  # [mu, sigma, lambda, delta]

# Define bounds to ensure parameters remain meaningful
param_bounds = [(None, None),  # mu (drift) can be any value
                (1e-4, None),  # sigma (volatility) must be positive
                (1e-4, None),  # lambda (jump intensity) must be positive
                (1e-4, None)]  # delta (jump size std dev) must be positive

# Perform MLE optimization
result_constrained = minimize(merton_log_likelihood, initial_params, args=(df['returns'],), 
                              method='L-BFGS-B', bounds=param_bounds)

# Extract estimated parameters
mu_hat, sigma_hat, lambda_hat, delta_hat = result_constrained.x

# Display estimated parameters
estimated_params_constrained = {
    "Mu (Drift)": mu_hat,
    "Sigma (Volatility)": sigma_hat,
    "Lambda (Jump Intensity)": lambda_hat,
    "Delta (Jump Size Std Dev)": delta_hat
}

print("Estimated Merton Jump-Diffusion Parameters for PREC:")
print(estimated_params_constrained)