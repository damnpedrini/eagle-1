#!/usr/bin/env python3
"""
EAGLE-1 Quick Demo - Demonstração rápida e funcional
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import re
from textblob import TextBlob

def calculate_rsi(prices, window=14):
    """Calcular RSI"""
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcular MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def get_fx_rate():
    """Buscar taxa USD/BRL"""
    try:
        response = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=BRL", timeout=5)
        data = response.json()
        return data['rates']['BRL']
    except:
        return 5.5

def get_crypto_news():
    """Buscar notícias de criptomoedas das últimas 24 horas"""
    try:
        # CoinDesk API para notícias recentes
        response = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            news = []
            current_time = datetime.now().timestamp()
            
            for article in data.get('Data', [])[:20]:  # Últimas 20 notícias
                # Verificar se é das últimas 24h
                article_time = article.get('published_on', 0)
                if current_time - article_time <= 86400:  # 24 horas em segundos
                    title = article.get('title', '')
                    body = article.get('body', '')[:200]  # Primeiros 200 chars
                    news.append(f"{title}. {body}")
            
            return news[:10]  # Retornar no máximo 10 notícias
        return []
    except:
        return []

def analyze_sentiment(news_list):
    """Analisar sentimento das notícias"""
    if not news_list:
        return 0.0, "Neutro"
    
    sentiments = []
    for news in news_list:
        try:
            blob = TextBlob(news)
            sentiments.append(blob.sentiment.polarity)
        except:
            continue
    
    if not sentiments:
        return 0.0, "Neutro"
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    if avg_sentiment > 0.1:
        sentiment_label = "Positivo"
    elif avg_sentiment < -0.1:
        sentiment_label = "Negativo"
    else:
        sentiment_label = "Neutro"
    
    return avg_sentiment, sentiment_label  

def main():
    print("🦅 EAGLE-1 v2.0 - Demonstração Rápida")
    print("=" * 45)
    
    try:
        print("📊 Buscando dados do Bitcoin (30 dias)...")
        data = yf.download("BTC-USD", period="30d", progress=False)
        
        if data.empty:
            print("❌ Erro ao buscar dados")
            return
        
        current_price = data['Close'].iloc[-1].item()
        print(f"💰 Preço atual: ${current_price:,.2f} USD")
        
        print("🔧 Calculando indicadores técnicos...")
        
        rsi = calculate_rsi(data['Close'])
        current_rsi = rsi.iloc[-1].item()
        print(f"📈 RSI (14): {current_rsi:.1f}")
        
        if current_rsi < 30:
            print("   🟢 Status: Oversold (oportunidade de compra)")
        elif current_rsi > 70:
            print("   🔴 Status: Overbought (possível correção)")
        else:
            print("   🟡 Status: Neutro")
        
        macd_line, signal_line = calculate_macd(data['Close'])
        current_macd = macd_line.iloc[-1].item()
        current_signal = signal_line.iloc[-1].item()
        
        print(f"📊 MACD: {current_macd:.2f}")
        if current_macd > current_signal:
            print("   🟢 Tendência: Bullish (alta)")
        else:
            print("   🔴 Tendência: Bearish (baixa)")
        
        # 3. Análise de tendência simples
        print("📈 Análise de tendência...")
        
        # Médias móveis
        sma_20 = data['Close'].rolling(20).mean().iloc[-1].item()
        
        print(f"   SMA 20: ${sma_20:.2f}")
        
        # Para 30 dias não temos dados suficientes para SMA 50, usar SMA 30
        if len(data) >= 30:
            sma_30 = data['Close'].rolling(30).mean().iloc[-1].item()
            print(f"   SMA 30: ${sma_30:.2f}")
        else:
            sma_30 = sma_20  # fallback
            print(f"   SMA 30: Dados insuficientes, usando SMA 20")
        
        if current_price > sma_20 > sma_30:
            trend_status = "🟢 Tendência de alta forte"
        elif current_price > sma_20:
            trend_status = "🟡 Tendência de alta moderada"
        elif current_price < sma_20 < sma_30:
            trend_status = "🔴 Tendência de baixa forte"
        else:
            trend_status = "🟡 Lateral"
        
        print(f"   Status: {trend_status}")
        
        # 4. Previsão simples baseada em tendência
        print("🔮 Previsão simples (5 dias)...")
        
        # Calcular mudança média dos últimos 5 dias
        recent_changes = data['Close'].pct_change().tail(5)
        avg_change = recent_changes.mean().item()
        
        # Projetar para 5 dias
        predicted_price = current_price * (1 + avg_change * 5)
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        print(f"📅 Previsão estimada (5 dias): ${predicted_price:,.2f} USD ({change_pct:+.1f}%)")
        
        if abs(change_pct) < 2:
            prediction_status = "➡️  Movimento lateral esperado"
        elif change_pct > 5:
            prediction_status = "🚀 Alta significativa esperada"
        elif change_pct > 0:
            prediction_status = "📈 Leve alta esperada"
        elif change_pct < -5:
            prediction_status = "📉 Queda significativa esperada"
        else:
            prediction_status = "📉 Leve queda esperada"
        
        print(f"   {prediction_status}")
        
        # 5. Conversão para BRL
        print("💱 Conversão para BRL...")
        fx_rate = get_fx_rate()
        price_brl = current_price * fx_rate
        pred_price_brl = predicted_price * fx_rate
        
        print(f"💰 Preço atual: R$ {price_brl:,.2f}")
        print(f"🔮 Previsão (5 dias): R$ {pred_price_brl:,.2f}")
        
        # 6. Análise de Sentimento das Notícias
        print("\n📰 Análise de Notícias (últimas 24h)...")
        news_list = get_crypto_news()
        
        if news_list:
            sentiment_score, sentiment_label = analyze_sentiment(news_list)
            print(f"📊 Sentimento geral: {sentiment_label} ({sentiment_score:.2f})")
            
            if sentiment_score > 0.1:
                sentiment_impact = "🟢 Notícias positivas podem impulsionar preço"
            elif sentiment_score < -0.1:
                sentiment_impact = "🔴 Notícias negativas podem pressionar preço"
            else:
                sentiment_impact = "🟡 Notícias neutras, baixo impacto esperado"
            
            print(f"   {sentiment_impact}")
            print(f"   Total de notícias analisadas: {len(news_list)}")
            
            # Mostrar algumas manchetes
            print("\n📄 Principais manchetes:")
            for i, news in enumerate(news_list[:3], 1):
                title = news.split('.')[0][:80] + "..." if len(news.split('.')[0]) > 80 else news.split('.')[0]
                print(f"   {i}. {title}")
        else:
            print("⚠️  Não foi possível obter notícias recentes")
            sentiment_score = 0
        
        # 7. Recomendação simples (incluindo sentimento)
        print("\n🎯 RECOMENDAÇÃO GERAL:")
        print("-" * 25)
        
        bullish_signals = 0
        bearish_signals = 0
        
        if current_rsi < 35:
            bullish_signals += 1
        elif current_rsi > 65:
            bearish_signals += 1
        
        if current_macd > current_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Adicionar sentimento das notícias na recomendação
        if 'sentiment_score' in locals():
            if sentiment_score > 0.1:
                bullish_signals += 1
            elif sentiment_score < -0.1:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            recommendation = "🟢 OTIMISTA - Considere compra"
        elif bearish_signals > bullish_signals:
            recommendation = "🔴 CAUTELOSO - Considere venda"
        else:
            recommendation = "🟡 NEUTRO - Aguarde melhor momento"
        
        print(f"{recommendation}")
        print(f"Sinais bullish: {bullish_signals} | Sinais bearish: {bearish_signals}")
        if 'sentiment_score' in locals():
            print(f"Sentimento incluído: {sentiment_label} ({sentiment_score:.2f})")
        
        # 8. Avisos
        print("\n⚠️  AVISOS IMPORTANTES:")
        print("• Este é um sistema educacional")
        print("• Não constitui aconselhamento financeiro")
        print("• Sempre faça sua própria pesquisa")
        print("• Use gestão de risco apropriada")
        
        print(f"\n✅ Análise concluída em {datetime.now().strftime('%H:%M:%S')}")
        print("🔍 Para análise mais avançada:")
        print("   python3 eagle1.py --help")
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")

if __name__ == "__main__":
    main()