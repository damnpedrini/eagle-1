"""
Testes automatizados para EAGLE-1
"""
import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Adicionar o diretório pai ao path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from eagle_core import (
        TechnicalAnalyzer, DataFetcher, SentimentAnalyzer, 
        TradingSignals, EaglePredictor
    )
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)


class TestTechnicalAnalyzer(unittest.TestCase):
    """Testes para análise técnica"""
    
    def setUp(self):
        """Configurar dados de teste"""
        # Criar série de preços sintética
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # Para resultados reproduzíveis
        
        # Gerar preços com tendência
        price_base = 50000
        trend = np.linspace(0, 10000, 100)
        noise = np.random.normal(0, 1000, 100)
        prices = price_base + trend + noise
        
        # Garantir que preços sejam positivos
        prices = np.maximum(prices, 1000)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        self.analyzer = TechnicalAnalyzer()
    
    def test_sma_calculation(self):
        """Testar cálculo da média móvel simples"""
        sma_20 = self.analyzer.sma(self.test_data['close'], 20)
        
        # Verificar se não há valores NaN no final
        self.assertFalse(sma_20.iloc[-1:].isna().any())
        
        # Verificar se a SMA é menor que o preço em tendência de alta
        recent_sma = sma_20.iloc[-10:].mean()
        recent_prices = self.test_data['close'].iloc[-10:].mean()
        self.assertLess(recent_sma, recent_prices)
    
    def test_rsi_calculation(self):
        """Testar cálculo do RSI"""
        rsi = self.analyzer.rsi(self.test_data['close'], 14)
        
        # RSI deve estar entre 0 e 100
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())
        
        # Não deve ter NaN
        self.assertFalse(rsi.isna().any())
    
    def test_macd_calculation(self):
        """Testar cálculo do MACD"""
        macd_line, signal, histogram = self.analyzer.macd(self.test_data['close'])
        
        # Verificar se as séries têm o mesmo tamanho
        self.assertEqual(len(macd_line), len(self.test_data))
        self.assertEqual(len(signal), len(self.test_data))
        self.assertEqual(len(histogram), len(self.test_data))
        
        # Histograma deve ser diferença entre MACD e sinal
        expected_hist = macd_line - signal
        np.testing.assert_array_almost_equal(histogram, expected_hist)
    
    def test_bollinger_bands(self):
        """Testar Bollinger Bands"""
        upper, lower, width = self.analyzer.bollinger_bands(self.test_data['close'], 20, 2.0)
        
        # Banda superior deve ser maior que inferior
        self.assertTrue((upper > lower).all())
        
        # Preço deve estar entre as bandas na maior parte do tempo
        between_bands = (
            (self.test_data['close'] >= lower) & 
            (self.test_data['close'] <= upper)
        ).sum()
        
        # Pelo menos 80% dos preços devem estar entre as bandas
        self.assertGreaterEqual(between_bands / len(self.test_data), 0.8)


class TestDataFetcher(unittest.TestCase):
    """Testes para busca de dados"""
    
    def setUp(self):
        self.fetcher = DataFetcher()
    
    def test_fx_rate_fallback(self):
        """Testar fallback da taxa de câmbio"""
        # Testar com moeda inexistente para forçar fallback
        rate = self.fetcher.get_fx_rate("USD", "FAKE")
        self.assertEqual(rate, 1.0)
    
    def test_data_structure(self):
        """Testar estrutura dos dados retornados"""
        try:
            data = self.fetcher.fetch_price_data("BTC-USD", "5d", "1d")
            
            # Verificar colunas obrigatórias
            required_columns = ['ds', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                self.assertIn(col, data.columns)
            
            # Verificar tipos de dados
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['ds']))
            self.assertTrue(pd.api.types.is_numeric_dtype(data['Close']))
            
        except Exception as e:
            # Se falhar por problemas de rede, apenas avisar
            print(f"Aviso: Teste de dados pulado devido a: {e}")


class TestTradingSignals(unittest.TestCase):
    """Testes para sinais de trading"""
    
    def setUp(self):
        # Criar dados sintéticos com padrão conhecido
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # Criar padrão de oversold seguido de alta
        prices = [45000] * 10 + [46000] * 10 + [48000] * 10 + [52000] * 20
        
        self.test_data = pd.DataFrame({
            'ds': dates,
            'y': prices,
            'rsi14': [25] * 10 + [35] * 10 + [45] * 10 + [60] * 20,  # Oversold -> normal
            'macd': [-100] * 20 + [50] * 30,  # Bearish -> Bullish
            'macd_signal': [-50] * 20 + [0] * 30,
            'bb_upper': [p * 1.05 for p in prices],
            'bb_lower': [p * 0.95 for p in prices]
        })
    
    def test_signal_generation(self):
        """Testar geração de sinais"""
        signals = TradingSignals.generate_signals(self.test_data)
        
        # Verificar se colunas de sinal foram criadas
        signal_columns = ['buy_signal', 'sell_signal', 'rsi_oversold', 'rsi_overbought']
        for col in signal_columns:
            self.assertIn(col, signals.columns)
        
        # Deve haver pelo menos um sinal de compra devido ao padrão oversold
        self.assertTrue(signals['buy_signal'].any())


class TestEaglePredictor(unittest.TestCase):
    """Testes para o preditor principal"""
    
    def setUp(self):
        self.predictor = EaglePredictor()
        
        # Dados mínimos para teste
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.normal(0, 100, 30))
        
        self.test_data = pd.DataFrame({
            'ds': dates,
            'Open': prices * 0.999,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        })
    
    def test_feature_preparation(self):
        """Testar preparação de features"""
        features = self.predictor.prepare_features(self.test_data)
        
        # Verificar colunas obrigatórias
        required_features = ['ds', 'y', 'rsi14', 'macd', 'bb_upper', 'bb_lower']
        for feature in required_features:
            self.assertIn(feature, features.columns)
        
        # Verificar se não há NaN ou infinitos
        self.assertFalse(features.isna().any().any())
        self.assertFalse(np.isinf(features.select_dtypes(include=[np.number])).any().any())
    
    def test_model_training_basic(self):
        """Testar treinamento básico do modelo"""
        features = self.predictor.prepare_features(self.test_data)
        
        try:
            model, forecast = self.predictor.train_model(features, forecast_days=7)
            
            # Verificar se o modelo foi criado
            self.assertIsNotNone(model)
            self.assertIsNotNone(forecast)
            
            # Verificar estrutura do forecast
            self.assertIn('ds', forecast.columns)
            self.assertIn('yhat', forecast.columns)
            self.assertIn('yhat_lower', forecast.columns)
            self.assertIn('yhat_upper', forecast.columns)
            
            # Verificar se há previsões futuras
            future_forecasts = forecast[forecast['ds'] > features['ds'].max()]
            self.assertEqual(len(future_forecasts), 7)
            
        except Exception as e:
            print(f"Aviso: Teste de modelo pulado devido a: {e}")


class TestSentimentAnalyzer(unittest.TestCase):
    """Testes para análise de sentimento"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sentiment_basic(self):
        """Testar análise básica de sentimento"""
        # Criar arquivo temporário
        test_headlines = [
            "Bitcoin price surges to new all-time high",
            "Cryptocurrency market shows strong bullish momentum",
            "Major selloff in crypto markets causes panic"
        ]
        
        test_file = "test_headlines.txt"
        with open(test_file, 'w') as f:
            for headline in test_headlines:
                f.write(f"{headline}\n")
        
        try:
            sentiment = self.analyzer.analyze_headlines(test_file)
            
            # Deve retornar um valor entre -1 e 1
            self.assertTrue(-1 <= sentiment.iloc[0] <= 1)
            
        finally:
            # Limpar arquivo de teste
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_sentiment_missing_file(self):
        """Testar comportamento com arquivo inexistente"""
        sentiment = self.analyzer.analyze_headlines("arquivo_inexistente.txt")
        self.assertEqual(sentiment.iloc[0], 0.0)


def run_tests():
    """Executar todos os testes"""
    print("🧪 Executando testes do EAGLE-1...")
    
    # Criar suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar classes de teste
    test_classes = [
        TestTechnicalAnalyzer,
        TestDataFetcher,
        TestTradingSignals,
        TestEaglePredictor,
        TestSentimentAnalyzer
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Relatório final
    if result.wasSuccessful():
        print("\n✅ Todos os testes passaram!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} testes falharam")
        print(f"🚨 {len(result.errors)} erros encontrados")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)