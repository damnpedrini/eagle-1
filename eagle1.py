#!/usr/bin/env python3
"""
EAGLE-1: Sistema Avançado de Previsão de Criptomoedas
Versão 2.0 - Sistema Profissional de Trading e Análise Técnica
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Importar módulos customizados
from eagle_core import EaglePredictor, DataFetcher, TradingSignals
from logger import log_info, log_error, log_warning


def create_sample_headlines():
    """Criar arquivo de exemplo de notícias"""
    sample_headlines = [
        "Bitcoin reaches new all-time high amid institutional adoption",
        "Major cryptocurrency exchange announces new security measures",
        "Federal Reserve considers digital currency regulations",
        "Tesla increases Bitcoin holdings in Q3 earnings report",
        "Cryptocurrency market shows strong bullish momentum",
        "New blockchain technology promises faster transactions",
        "Investment firm launches Bitcoin ETF for retail investors",
        "Crypto adoption grows in emerging markets worldwide",
        "Banking giant announces cryptocurrency trading services",
        "Regulatory clarity boosts investor confidence in digital assets"
    ]
    
    headlines_file = "sample_headlines.txt"
    with open(headlines_file, "w", encoding="utf-8") as f:
        for headline in sample_headlines:
            f.write(f"{headline}\n")
    
    log_info(f"Arquivo de exemplo criado: {headlines_file}")
    return headlines_file


def main():
    """Função principal do EAGLE-1"""
    parser = argparse.ArgumentParser(
        description="EAGLE-1 v2.0 - Sistema Avançado de Previsão de Criptomoedas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos básicos
    parser.add_argument("--symbol", default="BTC-USD", 
                       help="Símbolo da criptomoeda (ex: BTC-USD, ETH-USD)")
    parser.add_argument("--period", default="365d", 
                       help="Período histórico (ex: 365d, 2y, 5y)")
    parser.add_argument("--interval", default="1d", 
                       help="Intervalo dos dados (1d, 1h, 5m)")
    parser.add_argument("--forecast_days", type=int, default=30, 
                       help="Número de dias para previsão")
    
    # Análise de sentimento
    parser.add_argument("--headlines", default=None, 
                       help="Arquivo com notícias para análise de sentimento")
    parser.add_argument("--create_sample", action="store_true",
                       help="Criar arquivo de exemplo de notícias")
    
    # Configurações de saída
    parser.add_argument("--output_dir", default="outputs", 
                       help="Diretório de saída dos resultados")
    parser.add_argument("--output_prefix", default="eagle_analysis", 
                       help="Prefixo dos arquivos de saída")
    
    # Opções avançadas
    parser.add_argument("--no_fx", action="store_true", 
                       help="Não buscar taxa de câmbio USD/BRL")
    parser.add_argument("--interactive", action="store_true",
                       help="Criar gráfico interativo HTML")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Modo verboso (mais logs)")
    
    args = parser.parse_args()
    
    try:
        # Configurar diretório de saída
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        log_info("=== EAGLE-1 v2.0 INICIADO ===")
        log_info(f"Símbolo: {args.symbol}")
        log_info(f"Período: {args.period}")
        log_info(f"Previsão: {args.forecast_days} dias")
        
        # Criar arquivo de exemplo se solicitado
        if args.create_sample and not args.headlines:
            args.headlines = create_sample_headlines()
        
        # Inicializar sistema
        predictor = EaglePredictor()
        data_fetcher = DataFetcher()
        
        # 1. Buscar dados históricos
        log_info("📊 Buscando dados históricos...")
        historical_data = data_fetcher.fetch_price_data(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval
        )
        
        # 2. Preparar features técnicas
        log_info("🔧 Preparando indicadores técnicos...")
        features = predictor.prepare_features(
            historical_data, 
            headlines_file=args.headlines
        )
        
        # 3. Treinar modelo e fazer previsões
        log_info("🤖 Treinando modelo de machine learning...")
        model, forecast = predictor.train_model(
            features, 
            forecast_days=args.forecast_days
        )
        
        # 4. Buscar taxa de câmbio
        if not args.no_fx:
            log_info("💱 Buscando taxa de câmbio USD/BRL...")
            fx_rate = data_fetcher.get_fx_rate("USD", "BRL")
        else:
            fx_rate = 1.0
            log_info("💱 Usando taxa de câmbio padrão: 1.0")
        
        # 5. Gerar gráfico interativo
        if args.interactive:
            log_info("📈 Criando gráfico interativo...")
            output_html = output_dir / f"{args.output_prefix}_interactive.html"
            predictor.create_interactive_chart(
                forecast, features, str(output_html)
            )
        
        # 6. Gerar relatório completo
        log_info("📋 Gerando relatório de análise...")
        output_prefix = output_dir / args.output_prefix
        report = predictor.generate_report(
            forecast, features, fx_rate, str(output_prefix)
        )
        
        # 7. Mostrar resumo no terminal
        print("\n" + "="*60)
        print("🦅 EAGLE-1 - RESUMO DA ANÁLISE")
        print("="*60)
        
        current_price = features['y'].iloc[-1]
        future_forecast = forecast[forecast['ds'] > features['ds'].max()].head(30)
        
        print(f"💰 Preço Atual ({args.symbol}): ${current_price:,.2f} USD")
        
        if fx_rate != 1.0:
            print(f"💰 Preço em BRL: R$ {current_price * fx_rate:,.2f}")
        
        print(f"\n📅 PREVISÕES PARA OS PRÓXIMOS {args.forecast_days} DIAS:")
        
        # Mostrar previsões para próximos dias-chave
        key_days = [1, 7, 15, 30] if args.forecast_days >= 30 else [1, 7, args.forecast_days]
        
        for day in key_days:
            if day <= len(future_forecast):
                pred_price = future_forecast.iloc[day-1]['yhat']
                pred_lower = future_forecast.iloc[day-1]['yhat_lower'] 
                pred_upper = future_forecast.iloc[day-1]['yhat_upper']
                change_pct = ((pred_price - current_price) / current_price) * 100
                
                print(f"  📊 {day:2d} dia(s): ${pred_price:,.2f} USD ({change_pct:+.1f}%)")
                print(f"     Intervalo: ${pred_lower:,.2f} - ${pred_upper:,.2f} USD")
                
                if fx_rate != 1.0:
                    print(f"     Em BRL: R$ {pred_price * fx_rate:,.2f}")
                print()
        
        # Análise de indicadores atuais
        current_rsi = features['rsi14'].iloc[-1]
        current_macd = features['macd'].iloc[-1]
        current_macd_signal = features['macd_signal'].iloc[-1]
        
        print("📊 INDICADORES TÉCNICOS ATUAIS:")
        print(f"  RSI (14): {current_rsi:.1f}")
        
        if current_rsi < 30:
            print("    🟢 Status: Oversold (possível compra)")
        elif current_rsi > 70:
            print("    🔴 Status: Overbought (possível venda)")
        else:
            print("    🟡 Status: Neutro")
        
        print(f"  MACD: {current_macd:.4f}")
        if current_macd > current_macd_signal:
            print("    🟢 Status: Bullish (tendência de alta)")
        else:
            print("    🔴 Status: Bearish (tendência de baixa)")
        
        # Gerar sinais de trading
        signals = TradingSignals.generate_signals(features)
        recent_signals = signals.tail(10)
        
        buy_signals = recent_signals[recent_signals['buy_signal']].shape[0]
        sell_signals = recent_signals[recent_signals['sell_signal']].shape[0]
        
        print(f"\n🚨 SINAIS DE TRADING (últimos 10 dias):")
        print(f"  Sinais de Compra: {buy_signals}")
        print(f"  Sinais de Venda: {sell_signals}")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"  📈 Previsões: {output_prefix}_forecast.csv")
        print(f"  📋 Relatório: {output_prefix}_report.txt")
        
        if args.interactive:
            print(f"  🌐 Gráfico Interativo: {output_html}")
        
        print(f"\n✅ Análise concluída com sucesso!")
        print("="*60)
        
    except KeyboardInterrupt:
        log_warning("Análise interrompida pelo usuário")
        sys.exit(1)
        
    except Exception as e:
        log_error(f"Erro na execução: {str(e)}")
        print(f"\n❌ Erro: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()