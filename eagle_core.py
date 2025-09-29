"""
Classes principais do EAGLE-1: Sistema de Trading e Análise de Criptomoedas
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sentiment analysis
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    nltk.download("vader_lexicon")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

from logger import log_info, log_error, log_warning, log_debug, log_function_calls


class TechnicalAnalyzer:
    """Classe para análise técnica avançada"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series.fillna(50)
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalAnalyzer.ema(series, fast)
        ema_slow = TechnicalAnalyzer.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalyzer.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        ma = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std().fillna(0)
        upper = ma + n_std * std
        lower = ma - n_std * std
        width = (upper - lower) / ma.replace(0, np.nan)
        return upper, lower, width.fillna(0)
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
        return k_percent.fillna(50), d_percent.fillna(50)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return wr.fillna(-50)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        h_l = high - low
        h_c = np.abs(high - close.shift())
        l_c = np.abs(low - close.shift())
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()


class DataFetcher:
    """Classe para buscar dados de mercado"""
    
    @log_function_calls
    def fetch_price_data(self, symbol: str, period: str = "365d", interval: str = "1d") -> pd.DataFrame:
        """Buscar dados históricos de preços"""
        try:
            log_info(f"Buscando dados para {symbol} - período: {period}")
            
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                raise RuntimeError(f"Nenhum dado encontrado para {symbol}")
            
            data = data.rename_axis("ds").reset_index()
            data["ds"] = pd.to_datetime(data["ds"])
            data = data.sort_values("ds").reset_index(drop=True)
            
            log_info(f"Dados obtidos: {len(data)} registros de {data['ds'].min().date()} até {data['ds'].max().date()}")
            return data
            
        except Exception as e:
            log_error(f"Erro ao buscar dados: {str(e)}")
            raise
    
    @log_function_calls
    def get_fx_rate(self, base: str = "USD", target: str = "BRL") -> float:
        """Buscar taxa de câmbio"""
        try:
            log_info(f"Buscando taxa de câmbio {base}/{target}")
            
            url = f"https://api.exchangerate.host/latest?base={base}&symbols={target}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            rate = float(data["rates"][target])
            
            log_info(f"Taxa obtida: 1 {base} = {rate:.4f} {target}")
            return rate
            
        except Exception as e:
            log_warning(f"Erro ao buscar taxa de câmbio: {str(e)}. Usando taxa padrão 1.0")
            return 1.0


class SentimentAnalyzer:
    """Classe para análise de sentimento"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    @log_function_calls
    def analyze_headlines(self, headlines_file: str) -> pd.Series:
        """Analisar sentimento de notícias"""
        try:
            if not os.path.exists(headlines_file):
                log_warning(f"Arquivo de notícias não encontrado: {headlines_file}")
                return pd.Series([0.0])
            
            log_info(f"Analisando sentimento do arquivo: {headlines_file}")
            
            with open(headlines_file, 'r', encoding='utf-8') as f:
                headlines = f.readlines()
            
            sentiments = []
            for headline in headlines:
                headline = headline.strip()
                if headline:
                    score = self.analyzer.polarity_scores(headline)['compound']
                    sentiments.append(score)
            
            if not sentiments:
                return pd.Series([0.0])
            
            avg_sentiment = np.mean(sentiments)
            log_info(f"Sentimento médio: {avg_sentiment:.4f}")
            
            return pd.Series([avg_sentiment])
            
        except Exception as e:
            log_error(f"Erro na análise de sentimento: {str(e)}")
            return pd.Series([0.0])


class TradingSignals:
    """Classe para gerar sinais de trading"""
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Gerar sinais de compra/venda baseados em indicadores técnicos"""
        signals = df.copy()
        
        # Sinais baseados em RSI
        signals['rsi_oversold'] = signals['rsi14'] < 30
        signals['rsi_overbought'] = signals['rsi14'] > 70
        
        # Sinais baseados em MACD
        signals['macd_bullish'] = (signals['macd'] > signals['macd_signal']) & \
                                  (signals['macd'].shift(1) <= signals['macd_signal'].shift(1))
        signals['macd_bearish'] = (signals['macd'] < signals['macd_signal']) & \
                                  (signals['macd'].shift(1) >= signals['macd_signal'].shift(1))
        
        # Sinais baseados em Bollinger Bands
        signals['bb_buy'] = signals['y'] < signals['bb_lower']
        signals['bb_sell'] = signals['y'] > signals['bb_upper']
        
        # Sinal combinado (compra)
        signals['buy_signal'] = (signals['rsi_oversold'] | signals['bb_buy']) & signals['macd_bullish']
        
        # Sinal combinado (venda)
        signals['sell_signal'] = (signals['rsi_overbought'] | signals['bb_sell']) & signals['macd_bearish']
        
        return signals


class EaglePredictor:
    """Classe principal para previsão de preços"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model = None
        self.last_data = None
    
    @log_function_calls
    def prepare_features(self, df: pd.DataFrame, headlines_file: Optional[str] = None) -> pd.DataFrame:
        """Preparar features para o modelo"""
        try:
            log_info("Preparando features técnicas...")
            
            features = df.copy()
            features["y"] = features["Close"]
            
            # Médias móveis
            features["sma20"] = self.technical_analyzer.sma(features["y"], 20)
            features["sma50"] = self.technical_analyzer.sma(features["y"], 50)
            features["ema12"] = self.technical_analyzer.ema(features["y"], 12)
            features["ema26"] = self.technical_analyzer.ema(features["y"], 26)
            
            # Indicadores de momentum
            features["rsi14"] = self.technical_analyzer.rsi(features["y"], 14)
            
            # MACD
            macd_line, macd_signal, macd_hist = self.technical_analyzer.macd(features["y"])
            features["macd"] = macd_line
            features["macd_signal"] = macd_signal
            features["macd_hist"] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_width = self.technical_analyzer.bollinger_bands(features["y"])
            features["bb_upper"] = bb_upper
            features["bb_lower"] = bb_lower
            features["bb_width"] = bb_width
            
            # Stochastic
            stoch_k, stoch_d = self.technical_analyzer.stochastic(
                features["High"], features["Low"], features["Close"]
            )
            features["stoch_k"] = stoch_k
            features["stoch_d"] = stoch_d
            
            # Williams %R
            features["williams_r"] = self.technical_analyzer.williams_r(
                features["High"], features["Low"], features["Close"]
            )
            
            # ATR
            features["atr"] = self.technical_analyzer.atr(
                features["High"], features["Low"], features["Close"]
            )
            
            # Análise de volume
            if "Volume" in features.columns:
                features["volume_ma"] = features["Volume"].rolling(20).mean()
                features["volume_ratio"] = (features["Volume"] / features["volume_ma"]).fillna(1.0)
            else:
                features["volume_ratio"] = 1.0
            
            # Sentimento
            if headlines_file:
                sentiment = self.sentiment_analyzer.analyze_headlines(headlines_file)
                features["sentiment"] = sentiment.iloc[0] if len(sentiment) > 0 else 0.0
            else:
                features["sentiment"] = 0.0
            
            # Limpar dados
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill().fillna(0)
            
            # Selecionar colunas para o modelo
            model_features = [
                "ds", "y", "sma20", "sma50", "ema12", "ema26", "rsi14",
                "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_width",
                "stoch_k", "stoch_d", "williams_r", "atr", "volume_ratio", "sentiment"
            ]
            
            result = features[model_features].copy()
            log_info(f"Features preparadas: {len(result)} linhas, {len(model_features)} colunas")
            
            return result
            
        except Exception as e:
            log_error(f"Erro ao preparar features: {str(e)}")
            raise
    
    @log_function_calls
    def train_model(self, features_df: pd.DataFrame, forecast_days: int = 30) -> Tuple[Prophet, pd.DataFrame]:
        """Treinar modelo Prophet"""
        try:
            log_info("Treinando modelo Prophet...")
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Adicionar regressores
            regressors = [
                "sma20", "sma50", "ema12", "ema26", "rsi14", "macd", "macd_signal",
                "macd_hist", "bb_width", "stoch_k", "stoch_d", "williams_r", 
                "atr", "volume_ratio", "sentiment"
            ]
            
            for regressor in regressors:
                if regressor in features_df.columns:
                    model.add_regressor(regressor)
            
            # Treinar modelo
            model.fit(features_df[["ds", "y"] + regressors])
            
            # Criar previsões
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Preencher regressores para o futuro
            last_values = features_df.iloc[-1]
            for regressor in regressors:
                if regressor in features_df.columns:
                    future[regressor] = last_values[regressor]
            
            forecast = model.predict(future)
            
            self.model = model
            self.last_data = features_df
            
            log_info(f"Modelo treinado com sucesso. Previsões para {forecast_days} dias.")
            return model, forecast
            
        except Exception as e:
            log_error(f"Erro ao treinar modelo: {str(e)}")
            raise
    
    @log_function_calls
    def create_interactive_chart(self, forecast: pd.DataFrame, features_df: pd.DataFrame, 
                               output_file: str = "interactive_forecast.html"):
        """Criar gráfico interativo com Plotly"""
        try:
            log_info("Criando gráfico interativo...")
            
            # Criar subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Preço e Previsão', 'RSI', 'MACD', 'Volume'),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.2, 0.2, 0.1]
            )
            
            # Gráfico principal - Preço
            historical = features_df.tail(100)  # Últimos 100 dias
            future_forecast = forecast[forecast['ds'] > features_df['ds'].max()]
            
            # Preço histórico
            fig.add_trace(
                go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='lines',
                    name='Preço Histórico',
                    line=dict(color='blue')
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
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Banda de confiança
            fig.add_trace(
                go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Banda de Confiança',
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
                        line=dict(color='gray', dash='dot')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=historical['ds'],
                        y=historical['bb_lower'],
                        mode='lines',
                        name='BB Inferior',
                        line=dict(color='gray', dash='dot')
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
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                # Linhas de referência do RSI
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            if 'macd' in historical.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical['ds'],
                        y=historical['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=historical['ds'],
                        y=historical['macd_signal'],
                        mode='lines',
                        name='Sinal MACD',
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
            
            # Volume
            if 'volume_ratio' in historical.columns:
                fig.add_trace(
                    go.Bar(
                        x=historical['ds'],
                        y=historical['volume_ratio'],
                        name='Volume Ratio',
                        marker_color='lightblue'
                    ),
                    row=4, col=1
                )
            
            # Layout
            fig.update_layout(
                title="EAGLE-1 - Análise Técnica e Previsão de Preços",
                height=1000,
                showlegend=True,
                template="plotly_dark"
            )
            
            fig.update_xaxes(title_text="Data")
            fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            # Salvar
            fig.write_html(output_file)
            log_info(f"Gráfico interativo salvo em: {output_file}")
            
        except Exception as e:
            log_error(f"Erro ao criar gráfico interativo: {str(e)}")
    
    @log_function_calls
    def generate_report(self, forecast: pd.DataFrame, features_df: pd.DataFrame, 
                       fx_rate: float = 1.0, output_prefix: str = "eagle_report"):
        """Gerar relatório completo"""
        try:
            log_info("Gerando relatório completo...")
            
            # Dados de previsão
            future_forecast = forecast[forecast['ds'] > features_df['ds'].max()].head(30)
            
            # Converter para BRL se necessário
            if fx_rate != 1.0:
                future_forecast = future_forecast.copy()
                future_forecast['yhat_brl'] = future_forecast['yhat'] * fx_rate
                future_forecast['yhat_lower_brl'] = future_forecast['yhat_lower'] * fx_rate
                future_forecast['yhat_upper_brl'] = future_forecast['yhat_upper'] * fx_rate
            
            # Salvar previsões em CSV
            columns_to_save = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            if fx_rate != 1.0:
                columns_to_save.extend(['yhat_brl', 'yhat_lower_brl', 'yhat_upper_brl'])
            
            future_forecast[columns_to_save].to_csv(f"{output_prefix}_forecast.csv", index=False)
            
            # Gerar sinais de trading
            signals = TradingSignals.generate_signals(features_df)
            recent_signals = signals.tail(30)
            
            buy_signals = recent_signals[recent_signals['buy_signal']].shape[0]
            sell_signals = recent_signals[recent_signals['sell_signal']].shape[0]
            
            # Estatísticas atuais
            current_price = features_df['y'].iloc[-1]
            current_rsi = features_df['rsi14'].iloc[-1]
            
            # Previsão para próximos dias
            next_7_days = future_forecast.head(7)
            next_30_days = future_forecast.head(30)
            
            predicted_7d = next_7_days['yhat'].iloc[-1]
            predicted_30d = next_30_days['yhat'].iloc[-1]
            
            change_7d = ((predicted_7d - current_price) / current_price) * 100
            change_30d = ((predicted_30d - current_price) / current_price) * 100
            
            # Relatório em texto
            report = f"""
=== EAGLE-1 RELATÓRIO DE ANÁLISE ===
Data do Relatório: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

PREÇO ATUAL:
- Preço: ${current_price:,.2f} USD
- RSI: {current_rsi:.2f}

PREVISÕES:
- 7 dias: ${predicted_7d:,.2f} USD ({change_7d:+.2f}%)
- 30 dias: ${predicted_30d:,.2f} USD ({change_30d:+.2f}%)

SINAIS DE TRADING (últimos 30 dias):
- Sinais de Compra: {buy_signals}
- Sinais de Venda: {sell_signals}

RECOMENDAÇÃO GERAL:
"""
            
            if current_rsi < 30 and change_7d > 0:
                report += "🟢 COMPRA - RSI oversold com previsão positiva"
            elif current_rsi > 70 and change_7d < 0:
                report += "🔴 VENDA - RSI overbought com previsão negativa"
            elif change_30d > 10:
                report += "🟡 HOLD/COMPRA - Tendência de alta no médio prazo"
            elif change_30d < -10:
                report += "🟡 CAUTELA - Tendência de baixa no médio prazo"
            else:
                report += "🟡 NEUTRO - Mercado lateralizando"
            
            # Salvar relatório
            with open(f"{output_prefix}_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            
            log_info(f"Relatório salvo em: {output_prefix}_report.txt")
            log_info(f"Previsões salvas em: {output_prefix}_forecast.csv")
            
            return report
            
        except Exception as e:
            log_error(f"Erro ao gerar relatório: {str(e)}")
            raise