import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from scipy.stats import shapiro, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import StandardScaler

# -------------------------- Load Dataset --------------------------
# Define the file path for the meteorological dataset
file_path = "../datasets/CropSDEData/METEO_DEKADS_NUTS2_NL.csv"

# Read the dataset
df = pd.read_csv(file_path)

# Define the list of time series variables to analyze
time_series_cols = ['TMAX', 'TMIN', 'TAVG', 'VPRES', 'WSPD', 'PREC', 'ET0', 'RAD', 'RELH']

# Ensure only numerical columns are selected (exclude categorical columns like region names)
df_numeric = df.select_dtypes(include=[np.number])

# -------------------------- Apply Differencing for Stationarity --------------------------
# Create a copy of the dataset to apply transformations
diff_df = df_numeric.copy()

# Apply first-order differencing to remove trends
for col in time_series_cols:
    diff_df[col] = df_numeric[col].diff().dropna()

# Fill missing values with the median of each column to avoid NaNs in transformations
diff_df.fillna(diff_df.median(), inplace=True)

# -------------------------- Standardization --------------------------
# Standardize only numerical columns to prevent singular matrix issues
scaler = StandardScaler()
diff_df[time_series_cols] = scaler.fit_transform(diff_df[time_series_cols])

# -------------------------- Initialize Dictionary to Store Test Results --------------------------
results = {
    'ADF': {},          # Augmented Dickey-Fuller Test
    'KPSS': {},         # Kwiatkowski-Phillips-Schmidt-Shin Test
    'Shapiro-Wilk': {}, # Normality Test
    'Jarque-Bera': {},  # Normality Test
    'Ljung-Box': {},    # Autocorrelation Test
    'PACF': {}          # Partial Autocorrelation Function
}

# -------------------------- Stationarity Tests --------------------------
def adf_test(series):
    """
    Perform Augmented Dickey-Fuller (ADF) test to check stationarity.
    H0: Series is non-stationary (contains a unit root).
    If p-value < 0.05, we reject H0 and conclude the series is stationary.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Stationary': result[1] < 0.05}

def kpss_test(series):
    """
    Perform Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test to check stationarity.
    H0: Series is stationary.
    If p-value < 0.05, we reject H0 and conclude the series is non-stationary.
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    return {'KPSS Statistic': result[0], 'p-value': result[1], 'Stationary': result[1] > 0.05}

# Apply ADF and KPSS tests to each time series column
for col in time_series_cols:
    results['ADF'][col] = adf_test(diff_df[col])
    results['KPSS'][col] = kpss_test(diff_df[col])

# -------------------------- Normality Tests --------------------------
def shapiro_test(series):
    """
    Perform Shapiro-Wilk normality test.
    H0: Data follows a normal distribution.
    If p-value < 0.05, we reject H0 and conclude the data is not normally distributed.
    """
    result = shapiro(series.dropna())
    return {'Shapiro-Wilk Statistic': result[0], 'p-value': result[1], 'Normal': result[1] > 0.05}

def jarque_bera_test(series):
    """
    Perform Jarque-Bera test to check normality.
    H0: Data follows a normal distribution (zero skewness & kurtosis = 3).
    If p-value < 0.05, we reject H0 and conclude non-normality.
    """
    result = jarque_bera(series.dropna())
    return {'JB Statistic': result[0], 'p-value': result[1], 'Normal': result[1] > 0.05}

# Fix log transformation for PREC & WSPD to avoid NaN values
for col in ['PREC', 'WSPD']:
    if (diff_df[col] > 0).any():
        min_positive = diff_df[col][diff_df[col] > 0].min() * 0.1
    else:
        min_positive = 1e-5  # A small positive value to avoid log(0)
    diff_df[col] = np.log1p(diff_df[col].replace(0, min_positive)).fillna(diff_df[col].median())

# Apply normality tests to each time series column
for col in time_series_cols:
    results['Shapiro-Wilk'][col] = shapiro_test(diff_df[col])
    results['Jarque-Bera'][col] = jarque_bera_test(diff_df[col])

# -------------------------- Autocorrelation Tests --------------------------
def ljung_box_test(series, lags=10):
    """
    Perform Ljung-Box test to check for autocorrelation.
    H0: Data has no autocorrelation.
    If p-value < 0.05, we reject H0 and conclude that autocorrelation exists.
    """
    result = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    return {'LB Statistic': result['lb_stat'].values[0], 'p-value': result['lb_pvalue'].values[0], 'No Autocorrelation': result['lb_pvalue'].values[0] > 0.05}

# Fit AR(1) model to remove autocorrelation
residuals_df = diff_df.copy()
for col in time_series_cols:
    try:
        model = AutoReg(diff_df[col].dropna(), lags=1).fit()
        residuals_df[col] = model.resid  # Store residuals for further modeling
    except np.linalg.LinAlgError:
        residuals_df[col] += np.random.normal(0, 1e-6, len(residuals_df[col]))  # Add small noise to avoid singular matrix issues
    results['Ljung-Box'][col] = ljung_box_test(residuals_df[col])
    results['PACF'][col] = {'PACF_Lag1': pacf(residuals_df[col].dropna(), nlags=10)[1]}  # Check up to lag-10 for autocorrelation structure

# -------------------------- Convert Results to DataFrames --------------------------
adf_results = pd.DataFrame.from_dict(results['ADF'], orient='index')
kpss_results = pd.DataFrame.from_dict(results['KPSS'], orient='index')
shapiro_results = pd.DataFrame.from_dict(results['Shapiro-Wilk'], orient='index')
jarque_results = pd.DataFrame.from_dict(results['Jarque-Bera'], orient='index')
ljung_results = pd.DataFrame.from_dict(results['Ljung-Box'], orient='index')
pacf_results = pd.DataFrame.from_dict(results['PACF'], orient='index')

# -------------------------- Display Results --------------------------
print("Augmented Dickey-Fuller Test Results (After Differencing):")
print(adf_results)

print("\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results (After Differencing):")
print(kpss_results)

print("\nShapiro-Wilk Normality Test Results (After Log Transformations):")
print(shapiro_results)

print("\nJarque-Bera Normality Test Results (After Log Transformations):")
print(jarque_results)

print("\nLjung-Box Autocorrelation Test Results (After Removing Autocorrelation):")
print(ljung_results)

print("\nPartial Autocorrelation Function (PACF) Lag-1 to Lag-10 Values:")
print(pacf_results)
