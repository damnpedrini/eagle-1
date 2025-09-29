#!/usr/bin/env python3
"""
EAGLE-1 Demo - Demonstração rápida do sistema
"""

from eagle_core import DataFetcher, EaglePredictor
from logger import log_info

def main():
    print("🦅 EAGLE-1 v2.0 - Demonstração")
    print("=" * 40)
    
    # Inicializar sistema
    fetcher = DataFetcher()
    predictor = EaglePredictor()
    
    # 1. Buscar dados do Bitcoin
    print("📊 Buscando dados do Bitcoin...")
    data = fetcher.fetch_price_data("BTC-USD", "30d")
    current_price = data['Close'].iloc[-1]
    
    print(f"💰 Preço atual: ${current_price:,.2f} USD")
    
    # 2. Preparar indicadores
    print("🔧 Calculando indicadores técnicos...")
    features = predictor.prepare_features(data)
    
    rsi = features['rsi14'].iloc[-1]
    macd = features['macd'].iloc[-1]
    macd_signal = features['macd_signal'].iloc[-1]
    
    print(f"📈 RSI: {rsi:.1f}")
    
    if rsi < 30:
        rsi_status = "🟢 Oversold (oportunidade de compra)"
    elif rsi > 70:
        rsi_status = "🔴 Overbought (possível correção)"
    else:
        rsi_status = "🟡 Neutro"
    
    print(f"   Status: {rsi_status}")
    
    print(f"📊 MACD: {macd:.4f}")
    if macd > macd_signal:
        print("   Tendência: 🟢 Bullish (alta)")
    else:
        print("   Tendência: 🔴 Bearish (baixa)")
    
    # 3. Fazer previsão simples
    print("🤖 Gerando previsão para próximos 7 dias...")
    model, forecast = predictor.train_model(features, forecast_days=7)
    
    future_forecast = forecast[forecast['ds'] > features['ds'].max()]
    
    if len(future_forecast) > 0:
        pred_7d = future_forecast['yhat'].iloc[-1]
        change_pct = ((pred_7d - current_price) / current_price) * 100
        
        print(f"📅 Previsão 7 dias: ${pred_7d:,.2f} USD ({change_pct:+.1f}%)")
        
        if change_pct > 5:
            print("🚀 Tendência: Alta significativa")
        elif change_pct > 0:
            print("📈 Tendência: Leve alta") 
        elif change_pct < -5:
            print("📉 Tendência: Baixa significativa")
        else:
            print("➡️  Tendência: Lateral")
    
    # 4. Taxa de câmbio
    print("💱 Buscando taxa USD/BRL...")
    fx_rate = fetcher.get_fx_rate("USD", "BRL")
    print(f"💰 Preço em BRL: R$ {current_price * fx_rate:,.2f}")
    
    print("\n✅ Demonstração concluída!")
    print("🔍 Para análise completa, use: python3 eagle1.py --help")

if __name__ == "__main__":
    main()