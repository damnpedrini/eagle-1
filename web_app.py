"""
EAGLE-1 Web Interface
Interface web moderna para análise de criptomoedas usando Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Importar módulos customizados
try:
    from eagle_core import EaglePredictor, DataFetcher, TradingSignals
    from logger import log_info, log_error
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()


def configure_page():
    """Configurar página do Streamlit"""
    st.set_page_config(
        page_title="EAGLE-1 - Análise de Criptomoedas",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar():
    """Criar barra lateral com parâmetros"""
    st.sidebar.title("🦅 EAGLE-1")
    st.sidebar.markdown("### Sistema de Análise de Criptomoedas")
    
    # Parâmetros principais
    symbol = st.sidebar.selectbox(
        "Criptomoeda",
        ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD", "LINK-USD"],
        index=0
    )
    
    period = st.sidebar.selectbox(
        "Período Histórico",
        ["1y", "2y", "5y", "max"],
        index=0
    )
    
    forecast_days = st.sidebar.slider(
        "Dias para Previsão",
        min_value=7,
        max_value=90,
        value=30
    )
    
    # Opções avançadas
    st.sidebar.markdown("### Opções Avançadas")
    
    include_sentiment = st.sidebar.checkbox(
        "Incluir Análise de Sentimento",
        value=False,
        help="Usar notícias para análise de sentimento"
    )
    
    show_fx = st.sidebar.checkbox(
        "Mostrar Cotação em BRL",
        value=True,
        help="Converter valores para Real Brasileiro"
    )
    
    return {
        "symbol": symbol,
        "period": period,
        "forecast_days": forecast_days,
        "include_sentiment": include_sentiment,
        "show_fx": show_fx
    }


def display_current_metrics(features_df, fx_rate=1.0):
    """Exibir métricas atuais"""
    current_price = features_df['y'].iloc[-1]
    current_rsi = features_df['rsi14'].iloc[-1]
    current_macd = features_df['macd'].iloc[-1]
    current_macd_signal = features_df['macd_signal'].iloc[-1]
    
    # Cálculos de variação
    price_24h_ago = features_df['y'].iloc[-2] if len(features_df) > 1 else current_price
    price_change_24h = current_price - price_24h_ago
    price_change_pct = (price_change_24h / price_24h_ago) * 100 if price_24h_ago != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Preço Atual (USD)",
            f"${current_price:,.2f}",
            f"{price_change_pct:+.2f}%"
        )
        if fx_rate != 1.0:
            st.metric(
                "Preço Atual (BRL)",
                f"R$ {current_price * fx_rate:,.2f}",
                f"{price_change_pct:+.2f}%"
            )
    
    with col2:
        rsi_status = "🟢 Oversold" if current_rsi < 30 else "🔴 Overbought" if current_rsi > 70 else "🟡 Neutro"
        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_status)
    
    with col3:
        macd_trend = "🟢 Bullish" if current_macd > current_macd_signal else "🔴 Bearish"
        st.metric("MACD", f"{current_macd:.4f}", macd_trend)
    
    with col4:
        volume_current = features_df['volume_ratio'].iloc[-1] if 'volume_ratio' in features_df.columns else 0
        volume_status = "🟢 Alto" if volume_current > 1.2 else "🔴 Baixo" if volume_current < 0.8 else "🟡 Normal"
        st.metric("Volume", f"{volume_current:.2f}x", volume_status)


def create_price_chart(forecast, features_df):
    """Criar gráfico de preços e previsão"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Preço e Previsão', 'RSI', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Dados históricos (últimos 90 dias)
    historical = features_df.tail(90)
    future_forecast = forecast[forecast['ds'] > features_df['ds'].max()]
    
    # Preço histórico
    fig.add_trace(
        go.Scatter(
            x=historical['ds'],
            y=historical['y'],
            mode='lines',
            name='Preço Histórico',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Previsão
    fig.add_trace(
        go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines',
            name='Previsão',
            line=dict(color='#ff7f0e', dash='dash', width=2)
        ),
        row=1, col=1
    )
    
    # Banda de confiança
    fig.add_trace(
        go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Banda de Confiança',
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'bb_upper' in historical.columns:
        fig.add_trace(
            go.Scatter(
                x=historical['ds'],
                y=historical['bb_upper'],
                mode='lines',
                name='BB Superior',
                line=dict(color='gray', dash='dot', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical['ds'],
                y=historical['bb_lower'],
                mode='lines',
                name='BB Inferior',
                line=dict(color='gray', dash='dot', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi14' in historical.columns:
        fig.add_trace(
            go.Scatter(
                x=historical['ds'],
                y=historical['rsi14'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Linhas de referência RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # MACD
    if 'macd' in historical.columns:
        fig.add_trace(
            go.Scatter(
                x=historical['ds'],
                y=historical['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical['ds'],
                y=historical['macd_signal'],
                mode='lines',
                name='Sinal MACD',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Histograma MACD
        if 'macd_hist' in historical.columns:
            colors = ['green' if x >= 0 else 'red' for x in historical['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=historical['ds'],
                    y=historical['macd_hist'],
                    name='MACD Histograma',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Layout
    fig.update_layout(
        title="Análise Técnica Completa",
        height=800,
        showlegend=True,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def display_forecast_table(forecast, features_df, fx_rate=1.0):
    """Exibir tabela de previsões"""
    st.subheader("📅 Previsões Detalhadas")
    
    current_price = features_df['y'].iloc[-1]
    future_forecast = forecast[forecast['ds'] > features_df['ds'].max()].head(30)
    
    # Preparar dados da tabela
    table_data = []
    for idx, row in future_forecast.iterrows():
        change_pct = ((row['yhat'] - current_price) / current_price) * 100
        
        data_row = {
            "Data": row['ds'].strftime('%d/%m/%Y'),
            "Previsão (USD)": f"${row['yhat']:,.2f}",
            "Min (USD)": f"${row['yhat_lower']:,.2f}",
            "Max (USD)": f"${row['yhat_upper']:,.2f}",
            "Variação (%)": f"{change_pct:+.1f}%"
        }
        
        if fx_rate != 1.0:
            data_row["Previsão (BRL)"] = f"R$ {row['yhat'] * fx_rate:,.2f}"
        
        table_data.append(data_row)
    
    # Mostrar primeiros 7 dias
    df_display = pd.DataFrame(table_data[:7])
    st.dataframe(df_display, use_container_width=True)
    
    # Opção para mostrar mais
    if st.button("Ver Todos os 30 Dias"):
        df_full = pd.DataFrame(table_data)
        st.dataframe(df_full, use_container_width=True)


def display_trading_signals(features_df):
    """Exibir sinais de trading"""
    st.subheader("🚨 Sinais de Trading")
    
    # Gerar sinais
    signals = TradingSignals.generate_signals(features_df)
    recent_signals = signals.tail(30)
    
    buy_signals = recent_signals[recent_signals['buy_signal']].shape[0]
    sell_signals = recent_signals[recent_signals['sell_signal']].shape[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sinais de Compra (30d)", buy_signals)
    
    with col2:
        st.metric("Sinais de Venda (30d)", sell_signals)
    
    with col3:
        if buy_signals > sell_signals:
            sentiment = "🟢 Bullish"
        elif sell_signals > buy_signals:
            sentiment = "🔴 Bearish"
        else:
            sentiment = "🟡 Neutro"
        st.metric("Sentimento Geral", sentiment)
    
    # Últimos sinais
    last_signals = recent_signals.tail(10)
    signal_data = []
    
    for idx, row in last_signals.iterrows():
        if row['buy_signal']:
            signal_data.append({
                "Data": row['ds'].strftime('%d/%m/%Y'),
                "Sinal": "🟢 COMPRA",
                "Preço": f"${row['y']:,.2f}",
                "RSI": f"{row['rsi14']:.1f}"
            })
        elif row['sell_signal']:
            signal_data.append({
                "Data": row['ds'].strftime('%d/%m/%Y'),
                "Sinal": "🔴 VENDA",
                "Preço": f"${row['y']:,.2f}",
                "RSI": f"{row['rsi14']:.1f}"
            })
    
    if signal_data:
        st.write("#### Últimos Sinais:")
        df_signals = pd.DataFrame(signal_data)
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("Nenhum sinal de trading nos últimos 10 dias.")


def display_technical_summary(features_df):
    """Exibir resumo técnico"""
    st.subheader("📊 Resumo Técnico")
    
    current = features_df.iloc[-1]
    
    # Análise de tendência
    sma20 = current.get('sma20', 0)
    sma50 = current.get('sma50', 0)
    current_price = current['y']
    
    # Status de tendência
    if current_price > sma20 > sma50:
        trend = "🟢 Tendência de Alta Forte"
    elif current_price > sma20:
        trend = "🟡 Tendência de Alta Moderada"
    elif current_price < sma20 < sma50:
        trend = "🔴 Tendência de Baixa Forte"
    else:
        trend = "🟡 Tendência Lateral"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Análise de Tendência:**")
        st.write(trend)
        
        st.write("**Suporte e Resistência:**")
        bb_lower = current.get('bb_lower', current_price * 0.95)
        bb_upper = current.get('bb_upper', current_price * 1.05)
        st.write(f"Suporte: ${bb_lower:.2f}")
        st.write(f"Resistência: ${bb_upper:.2f}")
    
    with col2:
        st.write("**Indicadores de Momentum:**")
        rsi = current.get('rsi14', 50)
        if rsi < 30:
            rsi_status = "Oversold - Oportunidade de Compra"
        elif rsi > 70:
            rsi_status = "Overbought - Possível Correção"
        else:
            rsi_status = "Neutro"
        st.write(f"RSI: {rsi:.1f} - {rsi_status}")
        
        macd = current.get('macd', 0)
        macd_signal = current.get('macd_signal', 0)
        macd_status = "Bullish" if macd > macd_signal else "Bearish"
        st.write(f"MACD: {macd_status}")


def main():
    """Função principal da interface web"""
    configure_page()
    
    # Título principal
    st.title("🦅 EAGLE-1 - Sistema de Análise de Criptomoedas")
    st.markdown("### Análise técnica avançada e previsões com Machine Learning")
    
    # Barra lateral
    params = create_sidebar()
    
    # Botão principal
    if st.button("🚀 Executar Análise", type="primary"):
        
        with st.spinner("🔍 Analisando dados... Isso pode levar alguns minutos."):
            try:
                # Inicializar sistema
                predictor = EaglePredictor()
                data_fetcher = DataFetcher()
                
                # Progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Buscar dados
                status_text.text("📊 Buscando dados históricos...")
                progress_bar.progress(20)
                
                historical_data = data_fetcher.fetch_price_data(
                    symbol=params['symbol'],
                    period=params['period']
                )
                
                # 2. Preparar features
                status_text.text("🔧 Preparando indicadores técnicos...")
                progress_bar.progress(40)
                
                features = predictor.prepare_features(historical_data)
                
                # 3. Treinar modelo
                status_text.text("🤖 Treinando modelo de Machine Learning...")
                progress_bar.progress(60)
                
                model, forecast = predictor.train_model(
                    features, 
                    forecast_days=params['forecast_days']
                )
                
                # 4. Taxa de câmbio
                if params['show_fx']:
                    status_text.text("💱 Buscando taxa de câmbio...")
                    progress_bar.progress(80)
                    fx_rate = data_fetcher.get_fx_rate("USD", "BRL")
                else:
                    fx_rate = 1.0
                
                progress_bar.progress(100)
                status_text.text("✅ Análise concluída!")
                
                # Exibir resultados
                st.success("🎉 Análise realizada com sucesso!")
                
                # Métricas atuais
                display_current_metrics(features, fx_rate)
                
                # Gráfico principal
                st.plotly_chart(
                    create_price_chart(forecast, features),
                    use_container_width=True
                )
                
                # Tabs para diferentes análises
                tab1, tab2, tab3 = st.tabs(["📅 Previsões", "🚨 Sinais", "📊 Análise Técnica"])
                
                with tab1:
                    display_forecast_table(forecast, features, fx_rate)
                
                with tab2:
                    display_trading_signals(features)
                
                with tab3:
                    display_technical_summary(features)
                
                # Limpeza
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {str(e)}")
                log_error(f"Erro na interface web: {str(e)}")
    
    # Informações adicionais
    with st.expander("ℹ️ Sobre o EAGLE-1"):
        st.markdown("""
        **EAGLE-1** é um sistema avançado de análise de criptomoedas que combina:
        
        - 📈 **Análise Técnica Avançada**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR
        - 🤖 **Machine Learning**: Modelo Prophet do Facebook para previsões
        - 📰 **Análise de Sentimento**: Processamento de notícias (opcional)
        - 🎯 **Sinais de Trading**: Sinais automatizados de compra e venda
        - 📊 **Visualizações Interativas**: Gráficos dinâmicos com Plotly
        
        **Como usar:**
        1. Selecione a criptomoeda na barra lateral
        2. Configure o período histórico e dias de previsão
        3. Clique em "Executar Análise"
        4. Explore os resultados nas diferentes abas
        
        ⚠️ **Aviso**: Este sistema é apenas para fins educacionais. 
        Não constitui aconselhamento financeiro.
        """)


if __name__ == "__main__":
    main()