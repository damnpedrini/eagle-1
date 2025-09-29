#!/usr/bin/env python3
"""
Teste de Validação: O Prophet realmente prevê?
Vamos testar com dados históricos para ver se funcionaria no passado.
"""

import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_prophet_accuracy():
    print("🧪 TESTE DE VALIDAÇÃO DO PROPHET")
    print("=" * 40)
    print("Vamos testar se o Prophet funcionaria no passado...")
    print()
    
    # 1. Buscar dados históricos (90 dias atrás)
    print("📊 Buscando dados históricos...")
    end_date = datetime.now() - timedelta(days=7)  # 7 dias atrás
    start_date = end_date - timedelta(days=90)     # 90 dias antes disso
    
    # Dados para treinar (até 7 dias atrás)
    train_data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    
    # Dados reais dos últimos 7 dias (para comparar)
    test_data = yf.download("BTC-USD", period="7d", progress=False)
    
    print(f"✅ Dados de treino: {len(train_data)} dias")
    print(f"✅ Dados de teste: {len(test_data)} dias")
    
    # 2. Preparar dados para Prophet
    df_train = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data['Close'].values
    })
    
    # 3. Treinar modelo
    print("🤖 Treinando Prophet com dados até 7 dias atrás...")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(df_train)
    
    # 4. Fazer previsão para os últimos 7 dias
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    
    # 5. Comparar previsões com realidade
    print("\n🔍 COMPARANDO PREVISÕES COM REALIDADE:")
    print("-" * 50)
    
    last_train_price = float(train_data['Close'].iloc[-1])
    predictions = forecast.tail(7)
    
    errors = []
    
    for i in range(min(len(test_data), 7)):
        if i < len(predictions):
            predicted = predictions.iloc[i]['yhat']
            actual = float(test_data['Close'].iloc[i])
            error_pct = abs((predicted - actual) / actual) * 100
            errors.append(error_pct)
            
            print(f"Dia +{i+1}:")
            print(f"  Previsto: ${predicted:,.0f}")
            print(f"  Real:     ${actual:,.0f}")
            print(f"  Erro:     {error_pct:.1f}%")
            print()
    
    if errors:
        avg_error = np.mean(errors)
        print(f"📊 RESULTADO:")
        print(f"Erro médio: {avg_error:.1f}%")
        
        if avg_error < 5:
            print("🟢 MUITO BOM: Erro < 5%")
        elif avg_error < 10:
            print("🟡 RAZOÁVEL: Erro entre 5-10%")
        elif avg_error < 20:
            print("🟠 MÉDIO: Erro entre 10-20%")
        else:
            print("🔴 ALTO: Erro > 20%")
    
    print("\n💡 INTERPRETAÇÃO:")
    print("• Erros < 10% são considerados bons para crypto")
    print("• Prophet captura tendências, não eventos pontuais")
    print("• Funciona melhor em mercados com menos volatilidade")
    
    # 6. Teste de tendência
    if len(test_data) >= 2:
        real_trend = "alta" if test_data['Close'].iloc[-1] > test_data['Close'].iloc[0] else "baixa"
        pred_trend = "alta" if predictions.iloc[-1]['yhat'] > predictions.iloc[0]['yhat'] else "baixa"
        
        print(f"\n📈 TESTE DE TENDÊNCIA:")
        print(f"Tendência real: {real_trend}")
        print(f"Tendência prevista: {pred_trend}")
        
        if real_trend == pred_trend:
            print("✅ ACERTOU A DIREÇÃO!")
        else:
            print("❌ Errou a direção")

if __name__ == "__main__":
    try:
        test_prophet_accuracy()
        print("\n🎯 CONCLUSÃO:")
        print("O Prophet É um algoritmo real de ML que faz previsões baseadas")
        print("em padrões históricos. Não é mágica, mas é ciência de verdade!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        print("Isso pode acontecer por problemas de rede ou dados insuficientes")