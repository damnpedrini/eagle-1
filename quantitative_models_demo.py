#!/usr/bin/env python3
"""
Eagle-1 Quantitative Models Demo
Demonstração de todos os modelos financeiros avançados implementados
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from providers.binance_api import get_historical
from core.forecasting import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_quantitative_demo():
    print("🦅 Eagle-1 Quantitative Models Suite")
    print("=" * 50)
    
    # Obter dados do Bitcoin
    print("📈 Fetching Bitcoin data...")
    df = get_historical("BTCUSDT", limit=100)
    closes = df['close'].values
    current_price = closes[-1]
    returns = np.diff(np.log(closes))
    
    print(f"Current BTC Price: ${current_price:,.2f}")
    print(f"Historical Volatility: {np.std(returns) * np.sqrt(252):.2%}")
    print()
    
    # 1. Monte Carlo Simulation
    print("🎲 1. Monte Carlo Simulation (10,000 paths)")
    mc_result = monte_carlo_forecast(current_price, 0.8, 0.1, days=7, simulations=10000)
    print(f"   Expected Price (7 days): ${mc_result['mean']:,.2f}")
    print(f"   Standard Deviation: ${mc_result['std']:,.2f}")
    print(f"   95% Confidence Interval: [${mc_result['percentiles'][0]:,.2f}, ${mc_result['percentiles'][4]:,.2f}]")
    print()
    
    # 2. Geometric Brownian Motion
    print("📊 2. Geometric Brownian Motion")
    gbm_paths = geometric_brownian_motion(current_price, 0.1, 0.8, 7/365, 7, 1000)
    gbm_mean = np.mean(gbm_paths[:, -1])
    print(f"   Expected Price (7 days): ${gbm_mean:,.2f}")
    print(f"   Price Range: [${np.min(gbm_paths[:, -1]):,.2f}, ${np.max(gbm_paths[:, -1]):,.2f}]")
    print()
    
    # 3. Heston Model
    print("🌊 3. Heston Stochastic Volatility Model")
    heston = HestonModel(current_price, 0.64, 2.0, 0.64, 0.3, -0.5, 0.05)
    heston_S, heston_v = heston.simulate_paths(7/365, 7, 1000)
    heston_mean = np.mean(heston_S[:, -1])
    print(f"   Expected Price (7 days): ${heston_mean:,.2f}")
    print(f"   Final Volatility Range: [{np.min(np.sqrt(heston_v[:, -1])):.2%}, {np.max(np.sqrt(heston_v[:, -1])):.2%}]")
    print()
    
    # 4. SABR Model
    print("📈 4. SABR Model (Volatility Surface)")
    sabr = SABRModel(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    strikes = [current_price * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]
    print("   Implied Volatilities for different strikes:")
    for strike in strikes:
        iv = sabr.implied_volatility(current_price, strike, 7/365)
        print(f"   Strike ${strike:,.0f}: {iv:.2%}")
    print()
    
    # 5. Jump Diffusion (Merton)
    print("⚡ 5. Merton Jump Diffusion Model")
    jump_paths = merton_jump_diffusion(current_price, 0.1, 0.8, 0.1, 0, 0.2, 7/365, 7, 1000)
    jump_mean = np.mean(jump_paths[:, -1])
    print(f"   Expected Price (7 days): ${jump_mean:,.2f}")
    print(f"   Jump intensity: 0.1 jumps per day")
    print()
    
    # 6. Variance Gamma Model
    print("📊 6. Variance Gamma Model")
    vg_paths = variance_gamma_model(current_price, 0.1, 0.8, 0.2, 7/365, 7, 1000)
    vg_mean = np.mean(vg_paths[:, -1])
    print(f"   Expected Price (7 days): ${vg_mean:,.2f}")
    print()
    
    # 7. GARCH Model
    print("📈 7. GARCH(1,1) Volatility Model")
    try:
        garch = GARCHModel(returns[-50:])  # Usar últimos 50 retornos
        params = garch.fit_garch_11()
        vol_forecast = garch.forecast_volatility(params, 7)
        print(f"   GARCH Parameters: ω={params[0]:.6f}, α={params[1]:.3f}, β={params[2]:.3f}")
        print(f"   7-day Volatility Forecast: {np.mean(vol_forecast):.2%}")
    except Exception as e:
        print(f"   GARCH estimation error: {str(e)}")
    print()
    
    # 8. Binomial Tree
    print("🌳 8. Binomial Tree Model")
    binomial_prices = binomial_tree_forecast(current_price, 0.05, 0.8, 7/365, 7)
    binomial_mean = np.mean(binomial_prices)
    print(f"   Expected Price (7 days): ${binomial_mean:,.2f}")
    print(f"   Price nodes: {len(binomial_prices)} possible outcomes")
    print()
    
    # 9. Trinomial Tree
    print("🌲 9. Trinomial Tree Model")
    trinomial_prices = trinomial_tree_forecast(current_price, 0.05, 0.8, 7/365, 7)
    trinomial_mean = np.mean(trinomial_prices)
    print(f"   Expected Price (7 days): ${trinomial_mean:,.2f}")
    print(f"   Price nodes: {len(trinomial_prices)} possible outcomes")
    print()
    
    # 10. Black-Scholes PDE
    print("🔢 10. Black-Scholes PDE (Option Pricing)")
    K = current_price  # At-the-money
    T = 7/365
    call_price = black_scholes_pde_solution(current_price, K, T, 0.05, 0.8, 'call')
    put_price = black_scholes_pde_solution(current_price, K, T, 0.05, 0.8, 'put')
    print(f"   ATM Call Option (7 days): ${call_price:.2f}")
    print(f"   ATM Put Option (7 days): ${put_price:.2f}")
    print()
    
    # Ensemble Forecast
    all_forecasts = [mc_result['mean'], gbm_mean, heston_mean, jump_mean, vg_mean, binomial_mean, trinomial_mean]
    ensemble_mean = np.mean(all_forecasts)
    ensemble_std = np.std(all_forecasts)
    
    print("🎯 ENSEMBLE FORECAST SUMMARY")
    print("=" * 40)
    print(f"Ensemble Mean: ${ensemble_mean:,.2f}")
    print(f"Model Agreement (Std Dev): ${ensemble_std:,.2f}")
    print(f"Confidence: {max(0, 100 - (ensemble_std/ensemble_mean)*100):.1f}%")
    print(f"Expected Return: {((ensemble_mean/current_price - 1)*100):+.2f}%")
    
    # Plotar histograma das previsões
    plt.figure(figsize=(12, 8), facecolor='black')
    
    # Subplot 1: Monte Carlo histogram
    plt.subplot(2, 2, 1)
    plt.hist(mc_result['all_prices'], bins=50, alpha=0.7, color='orange', edgecolor='white')
    plt.axvline(mc_result['mean'], color='red', linestyle='--', label=f'Mean: ${mc_result["mean"]:.0f}')
    plt.title('Monte Carlo Distribution', color='white')
    plt.xlabel('Price (USD)', color='white')
    plt.ylabel('Frequency', color='white')
    plt.legend()
    plt.gca().set_facecolor('black')
    
    # Subplot 2: Model comparison
    plt.subplot(2, 2, 2)
    models = ['MC', 'GBM', 'Heston', 'Jump', 'VG', 'Binomial', 'Trinomial']
    plt.bar(models, all_forecasts, color='orange', alpha=0.7, edgecolor='white')
    plt.axhline(ensemble_mean, color='red', linestyle='--', label=f'Ensemble: ${ensemble_mean:.0f}')
    plt.title('Model Comparison', color='white')
    plt.ylabel('Price (USD)', color='white')
    plt.legend()
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('black')
    
    # Subplot 3: Heston paths (sample)
    plt.subplot(2, 2, 3)
    for i in range(min(100, heston_S.shape[0])):
        plt.plot(heston_S[i, :], alpha=0.1, color='orange')
    plt.plot(np.mean(heston_S, axis=0), color='red', linewidth=2, label='Mean Path')
    plt.title('Heston Model Paths', color='white')
    plt.xlabel('Days', color='white')
    plt.ylabel('Price (USD)', color='white')
    plt.legend()
    plt.gca().set_facecolor('black')
    
    # Subplot 4: Volatility surface (SABR)
    plt.subplot(2, 2, 4)
    strikes_range = np.linspace(current_price*0.8, current_price*1.2, 20)
    times_range = np.linspace(1/365, 30/365, 10)
    vol_surface = np.zeros((len(times_range), len(strikes_range)))
    
    for i, T in enumerate(times_range):
        for j, K in enumerate(strikes_range):
            try:
                vol_surface[i, j] = sabr.implied_volatility(current_price, K, T)
            except:
                vol_surface[i, j] = 0.5
    
    plt.contourf(strikes_range, times_range*365, vol_surface, levels=20, cmap='plasma')
    plt.colorbar(label='Implied Volatility')
    plt.title('SABR Volatility Surface', color='white')
    plt.xlabel('Strike (USD)', color='white')
    plt.ylabel('Time to Expiry (days)', color='white')
    plt.gca().set_facecolor('black')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_quantitative_demo()