"""
test_finance_metrics.py
=======================
Unit tests for finance_metrics.py calculations:
- Annualized Return
- Annualized Volatility
- Sharpe Ratio (7% Rf)
- Maximum Drawdown (MDD)
- CAPM Beta
- Correlation Matrix
"""

import unittest
import pandas as pd
import numpy as np

from finance_metrics import (
    calculate_daily_returns,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_drawdown,
    calculate_beta,
    calculate_correlation_matrix,
    get_full_risk_return_profile,
)


class TestFinanceMetrics(unittest.TestCase):

    def test_calculate_daily_returns(self):
        prices = pd.Series([100.0, 110.0, 99.0])
        returns = calculate_daily_returns(prices)
        self.assertEqual(len(returns), 3)
        self.assertTrue(np.isnan(returns.iloc[0]))
        self.assertAlmostEqual(returns.iloc[1], 10.0, places=2)
        self.assertAlmostEqual(returns.iloc[2], -10.0, places=2)

    def test_calculate_annualized_return(self):
        # 253 trading days (252 steps) doubling from 100 to 200 -> Exactly 100% CAGR
        prices = pd.Series(np.linspace(100.0, 200.0, 253))
        cagr = calculate_annualized_return(prices, periods_per_year=252)
        self.assertAlmostEqual(cagr, 100.0, places=1)

        # Single price edge case
        self.assertEqual(calculate_annualized_return(pd.Series([100.0])), 0.0)

    def test_calculate_annualized_volatility(self):
        # Constant returns should have zero volatility
        flat_returns = pd.Series([1.0] * 100)
        self.assertEqual(calculate_annualized_volatility(flat_returns), 0.0)

        # Alternating returns with known std
        returns = pd.Series([2.0, -2.0, 2.0, -2.0] * 50)
        vol = calculate_annualized_volatility(returns, periods_per_year=252)
        self.assertGreater(vol, 0.0)

    def test_calculate_sharpe_ratio(self):
        # Annual return = 25%, Volatility = 18%, Rf = 7% -> (25 - 7) / 18 = 1.0
        sharpe = calculate_sharpe_ratio(25.0, 18.0, risk_free_rate_pct=7.0)
        self.assertEqual(sharpe, 1.0)

        # Annual return = 7% (equal to Rf) -> Sharpe = 0.0
        self.assertEqual(calculate_sharpe_ratio(7.0, 15.0, risk_free_rate_pct=7.0), 0.0)

        # Zero volatility edge case
        self.assertEqual(calculate_sharpe_ratio(15.0, 0.0), 0.0)

    def test_calculate_drawdown(self):
        # Peak 100 -> 120 -> drops to 90 -> recovery to 110
        # Peak was 120, trough was 90. Max drawdown = (90 - 120) / 120 = -25.0%
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])
        dd_series, max_dd = calculate_drawdown(prices)
        self.assertEqual(max_dd, -25.0)
        self.assertEqual(dd_series.iloc[1], 0.0)  # at peak, drawdown is 0%
        self.assertEqual(dd_series.iloc[2], -25.0)

    def test_calculate_beta(self):
        np.random.seed(42)
        bench_ret = pd.Series(np.random.normal(0.05, 1.0, 100))
        # Stock with exact 2x sensitivity
        stock_ret_2x = bench_ret * 2.0
        beta_2x = calculate_beta(stock_ret_2x, bench_ret)
        self.assertAlmostEqual(beta_2x, 2.0, places=2)

        # Stock identical to benchmark
        beta_1x = calculate_beta(bench_ret, bench_ret)
        self.assertAlmostEqual(beta_1x, 1.0, places=2)

    def test_calculate_correlation_matrix(self):
        # Create prices where Stock_A and Stock_B have identical returns (+5%, -3%, +4%)
        # and Stock_C has opposite returns (-5%, +3%, -4%)
        returns_a = [0.05, -0.03, 0.04, -0.02, 0.06]
        returns_b = [0.05, -0.03, 0.04, -0.02, 0.06]
        returns_c = [-0.05, 0.03, -0.04, 0.02, -0.06]

        price_a = 100.0 * np.cumprod([1.0] + [1.0 + r for r in returns_a])
        price_b = 50.0 * np.cumprod([1.0] + [1.0 + r for r in returns_b])
        price_c = 200.0 * np.cumprod([1.0] + [1.0 + r for r in returns_c])

        df = pd.DataFrame({
            "Stock_A": price_a,
            "Stock_B": price_b,
            "Stock_C": price_c,
        })
        corr = calculate_correlation_matrix(df)
        self.assertEqual(corr.shape, (3, 3))
        self.assertAlmostEqual(corr.loc["Stock_A", "Stock_B"], 1.0, places=1)
        self.assertAlmostEqual(corr.loc["Stock_A", "Stock_C"], -1.0, places=1)

    def test_get_full_risk_return_profile(self):
        prices = pd.Series([100.0, 105.0, 102.0, 110.0, 115.0] * 50)
        profile = get_full_risk_return_profile(prices, risk_free_rate_pct=7.0)

        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "drawdown_series",
            "beta",
        }
        self.assertTrue(expected_keys.issubset(profile.keys()))
        self.assertIsInstance(profile["drawdown_series"], pd.Series)


if __name__ == "__main__":
    unittest.main()
