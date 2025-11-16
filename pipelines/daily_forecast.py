from providers.binance_api import get_historical
from core.forecasting import (moving_average, calculate_volatility, black_scholes_multi_day_forecast, 
                             advanced_forecast_suite, monte_carlo_forecast)
import mplfinance as mpf

def run_daily_forecast(symbol="BTC", days=30):
    import pandas as pd
    import numpy as np
    pair = f"{symbol.upper()}USDT"

    df = get_historical(pair)

    print(f"\nBTC Advanced Quantitative Models Forecast (next 7 days):")
    import matplotlib.pyplot as plt
    import matplotlib
    # Configurar backend do matplotlib
    try:
        matplotlib.use('TkAgg')  # Tenta usar TkAgg primeiro
    except:
        try:
            matplotlib.use('Qt5Agg')  # Fallback para Qt5Agg
        except:
            pass  # Usa o backend padrão
    
    closes = df['close'].values
    current_price = closes[-1]
    
    # Executa todos os modelos avançados
    print("Running advanced quantitative models...")
    results = advanced_forecast_suite(closes, current_price, days=7)
    
    # Calcula volatilidade histórica
    volatility = calculate_volatility(closes, window=30)
    print(f"Historical volatility: {volatility:.2%}")
    
    last_date = df.index[-1]
    forecast_dates = [(last_date + pd.Timedelta(days=i)).strftime('%b %d') for i in range(1, 8)]
    
    # Gera previsões dia a dia usando Black-Scholes
    daily_forecasts = black_scholes_multi_day_forecast(current_price, volatility, days=7)
    
    # Usa ensemble de TODOS os modelos para cada dia
    ensemble_base = (results['monte_carlo']['mean'] + results['gbm_mean'] + results['heston_mean'] + 
                    results['jump_mean'] + results['vg_mean'] + results['binomial_mean'] + 
                    results['trinomial_mean']) / 7
    
    # Aplica variação diária baseada no Black-Scholes
    forecast_values = []
    for i, daily_price in enumerate(daily_forecasts):
        # Combina ensemble com variação diária
        adjustment_factor = daily_price / current_price
        daily_ensemble = ensemble_base * adjustment_factor
        forecast_values.append(daily_ensemble)
    
    # Mostra resultados de cada modelo
    print(f"\n📊 ALL QUANTITATIVE MODELS Results (7-day forecast):")
    print("=" * 60)
    print(f"🎲 Monte Carlo (10k sims): {results['monte_carlo']['mean']:.2f} USD")
    print(f"📊 Geometric Brownian Motion: {results['gbm_mean']:.2f} USD")
    print(f"🌊 Heston Stochastic Vol: {results['heston_mean']:.2f} USD")
    print(f"⚡ Jump Diffusion (Merton): {results['jump_mean']:.2f} USD")
    print(f"📈 Variance Gamma: {results['vg_mean']:.2f} USD")
    print(f"🌳 Binomial Tree: {results['binomial_mean']:.2f} USD")
    print(f"🌲 Trinomial Tree: {results['trinomial_mean']:.2f} USD")
    print(f"📈 GARCH Vol Forecast: {np.mean(results['garch_volatility']):.2%}")
    print(f"🔢 SABR Implied Vol: {results['sabr_iv']:.2%}")
    print(f"🎯 Black-Scholes Call: ${results['bs_call_price']:.2f}")
    print("=" * 60)
    print(f"🎯 ENSEMBLE AVERAGE: {ensemble_base:.2f} USD")

    plt.figure(figsize=(10,5), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.plot(forecast_dates, forecast_values, color='#ff5a1f', linewidth=2)

    # Remove spines, ticks, grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='white', which='both')
    plt.xticks(color='white', fontsize=12)
    plt.yticks(color='white', fontsize=12)
    plt.title('Eagle-1 BTC QUANTITATIVE MODELS ENSEMBLE USD', color='white', fontsize=18, fontweight='bold')
    plt.ylabel('Ensemble Forecast Price (USD)', color='white', fontsize=14)

    # Annotate forecast values
    for i, (x, y) in enumerate(zip(forecast_dates, forecast_values)):
        plt.text(i, y, f'${y:.0f}', color='white', fontsize=11, ha='center', va='bottom', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', fc='#222', ec='white', alpha=0.8))
    
    # Adicionar pontos no gráfico para destacar os valores
    plt.scatter(range(len(forecast_dates)), forecast_values, color='#ff5a1f', s=50, zorder=5)

    plt.tight_layout()
    print('\n📈 Daily Ensemble Forecast for next 7 days:')
    print('=' * 45)
    for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
        change = ((price / current_price - 1) * 100)
        print(f'Day {i+1} ({date}): ${price:.2f} USD ({change:+.1f}%)')
    
    # Mostrar estatísticas do Monte Carlo
    print(f"\n📊 Monte Carlo Statistics:")
    print(f"Mean: {results['monte_carlo']['mean']:.2f} USD")
    print(f"Std Dev: {results['monte_carlo']['std']:.2f} USD")
    print(f"95% Confidence Interval: [{results['monte_carlo']['percentiles'][0]:.2f}, {results['monte_carlo']['percentiles'][4]:.2f}] USD")
    
    # Mostrar volatilidade GARCH
    print(f"\n📈 GARCH Volatility Forecast (7 days): {np.mean(results['garch_volatility']):.2%}")
    
    # Garantir que o gráfico apareça
    print("\n🎯 Displaying chart...")
    plt.ion()  # Modo interativo
    plt.show()
    plt.pause(0.1)  # Pequena pausa para garantir renderização
    
    # Salvar gráfico como imagem também
    try:
        plt.savefig('eagle1_forecast.png', facecolor='black', dpi=150, bbox_inches='tight')
        print("📊 Chart saved as 'eagle1_forecast.png'")
    except:
        print("⚠️  Could not save chart image")
    
    input("Press Enter to continue...")  # Manter gráfico aberto
