"""
finance_metrics.py
==================
Quantitative financial metrics module for stock market intelligence.
Provides standalone, mathematically rigorous calculations for:
- Daily and annualized returns (CAGR)
- Annualized volatility
- Sharpe ratio (with customizable risk-free rate, default 7% for Indian markets)
- Maximum Drawdown (MDD) and drawdown time series
- Capital Asset Pricing Model (CAPM) Beta versus benchmark
- Multi-asset correlation matrix
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Computes percentage daily returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Series of historical closing prices.

    Returns
    -------
    pd.Series
        Daily percentage returns (e.g., 1.5 for +1.5%).
    """
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float)
    return prices.pct_change() * 100.0


def calculate_annualized_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR) from a historical price series.
    Formula: CAGR = (P_end / P_start) ** (periods_per_year / N) - 1

    If the period has fewer than 252 trading days, the metric reflects the
    annualized compounding rate over the sample window.

    Parameters
    ----------
    prices : pd.Series
        Series of historical closing prices sorted chronologically.
    periods_per_year : int, default 252
        Number of trading sessions in a year (NSE standard: 252).

    Returns
    -------
    float
        Annualized return expressed as a percentage (e.g., 18.5 for 18.5%).
        Returns 0.0 if prices series is invalid or contains non-positive values.
    """
    clean_prices = prices.dropna()
    n_periods = len(clean_prices)
    if n_periods < 2:
        return 0.0

    p_start = float(clean_prices.iloc[0])
    p_end = float(clean_prices.iloc[-1])

    if p_start <= 0 or p_end <= 0:
        return 0.0

    # Compound Annual Growth Rate (CAGR)
    cagr = (p_end / p_start) ** (periods_per_year / (n_periods - 1)) - 1.0
    return round(cagr * 100.0, 2)


def calculate_annualized_volatility(daily_returns_pct: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates annualized volatility from daily percentage returns.
    Formula: Volatility_ann = Std(Daily_Returns) * sqrt(periods_per_year)

    Parameters
    ----------
    daily_returns_pct : pd.Series
        Series of daily returns in percent (e.g., +1.2, -0.8).
    periods_per_year : int, default 252
        Number of trading days per year (252 for NSE India).

    Returns
    -------
    float
        Annualized volatility in percent (e.g., 21.4 for 21.4%).
        Returns 0.0 if sample size is insufficient.
    """
    clean_returns = daily_returns_pct.dropna()
    if len(clean_returns) < 2:
        return 0.0

    daily_std = clean_returns.std()
    ann_vol = daily_std * np.sqrt(periods_per_year)
    return round(float(ann_vol), 2)


def calculate_sharpe_ratio(
    annualized_return_pct: float,
    annualized_volatility_pct: float,
    risk_free_rate_pct: float = 7.0
) -> float:
    """
    Calculates the Sharpe Ratio of an investment.
    Formula: Sharpe = (Annualized_Return - Risk_Free_Rate) / Annualized_Volatility

    Parameters
    ----------
    annualized_return_pct : float
        Annualized return in percent (e.g., 15.0).
    annualized_volatility_pct : float
        Annualized volatility in percent (e.g., 18.0).
    risk_free_rate_pct : float, default 7.0
        Benchmark risk-free hurdle in percent (7.0% corresponds to India 10Y G-Sec / RBI repo).

    Returns
    -------
    float
        Sharpe ratio rounded to 2 decimal places.
        Returns 0.0 if volatility is zero or negative.
    """
    if annualized_volatility_pct <= 0:
        return 0.0

    excess_return = annualized_return_pct - risk_free_rate_pct
    sharpe = excess_return / annualized_volatility_pct
    return round(float(sharpe), 2)


def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float]:
    """
    Calculates the cumulative drawdown series and the Maximum Drawdown (MDD).
    Formula:
        Peak_t = max(Price_0, ..., Price_t)
        Drawdown_t = (Price_t - Peak_t) / Peak_t

    Parameters
    ----------
    prices : pd.Series
        Chronological series of prices.

    Returns
    -------
    Tuple[pd.Series, float]
        - pd.Series: Drawdown percentage series (values <= 0.0, e.g., -5.2%).
        - float: Maximum drawdown percentage (e.g., -24.8%).
    """
    clean_prices = prices.dropna()
    if clean_prices.empty:
        return pd.Series(dtype=float), 0.0

    rolling_peak = clean_prices.cummax()
    drawdown_pct = ((clean_prices - rolling_peak) / rolling_peak) * 100.0
    max_drawdown = float(drawdown_pct.min()) if not drawdown_pct.empty else 0.0

    return drawdown_pct, round(max_drawdown, 2)


def calculate_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculates the CAPM Beta of a stock relative to a benchmark index (e.g., Nifty 50).
    Formula: Beta = Covariance(R_stock, R_benchmark) / Variance(R_benchmark)

    Parameters
    ----------
    stock_returns : pd.Series
        Daily percentage returns of the stock.
    benchmark_returns : pd.Series
        Daily percentage returns of the benchmark index.

    Returns
    -------
    float
        Beta coefficient rounded to 2 decimal places. Returns 1.0 if variance is zero.
    """
    combined = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(combined) < 5:
        return 1.0

    s = combined.iloc[:, 0]
    b = combined.iloc[:, 1]

    cov = np.cov(s, b)[0, 1]
    var_bench = np.var(b, ddof=1)

    if var_bench <= 0 or np.isnan(var_bench):
        return 1.0

    beta = cov / var_bench
    return round(float(beta), 2)


def calculate_correlation_matrix(pivot_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix of daily percentage returns across multiple stocks.

    Parameters
    ----------
    pivot_prices : pd.DataFrame
        DataFrame with Date as index, stock names as columns, and Close prices as values.

    Returns
    -------
    pd.DataFrame
        Correlation matrix bounded between -1.0 and 1.0.
    """
    if pivot_prices.empty:
        return pd.DataFrame()
    daily_returns = pivot_prices.pct_change().dropna(how="all")
    corr = daily_returns.corr().round(2)
    return corr


def get_full_risk_return_profile(
    prices: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate_pct: float = 7.0
) -> Dict[str, Any]:
    """
    Comprehensive helper that calculates all key risk-return statistics for a single asset.

    Returns
    -------
    dict with keys:
        - annualized_return
        - annualized_volatility
        - sharpe_ratio
        - max_drawdown
        - drawdown_series
        - beta (if benchmark provided)
    """
    daily_ret = calculate_daily_returns(prices)
    ann_ret = calculate_annualized_return(prices)
    ann_vol = calculate_annualized_volatility(daily_ret)
    sharpe = calculate_sharpe_ratio(ann_ret, ann_vol, risk_free_rate_pct)
    dd_series, max_dd = calculate_drawdown(prices)

    beta = 1.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        beta = calculate_beta(daily_ret, benchmark_returns)

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "drawdown_series": dd_series,
        "beta": beta
    }
