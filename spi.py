from numpy import log as ln
from scipy.stats import gamma, norm
import pandas as pd

def spi(df, col, scale):
    # Mean of Precipitation
    df_mean = df[col].mean()
    # Standardized Precipitation
    df['precip_std'] = df[col].std()

    # Scale of SPI
    f = scale
    rolling = df[col].rolling(f, min_periods=f).mean()

    # obtain ln(x_m) and, parameter A
    df['ln(x_m)'] = ln(rolling)
    contador = len(df['ln(x_m)'].dropna())
    soma_ln = df['ln(x_m)'].sum()
    A = ln(df_mean) - (soma_ln/contador)

    # obtain shape and scale
    alpha = (1/(4*A))*((1+(4*A)/(3))**(1/2))
    beta  = df_mean/alpha

    # obtain gamma_1
    gamma_1 = gamma.sf(x=rolling,a=alpha, scale=beta)
    df[f'spi_gamma_{f}'] = pd.DataFrame(norm.ppf(gamma_1), columns=['spi'])
    df_new = df[['time',f'spi_gamma_{f}']]
    return df_new