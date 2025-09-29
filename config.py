"""
Configurações do EAGLE-1
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configurações da aplicação"""
    
    # API Settings
    ALPHA_VANTAGE_API_KEY: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: str = Field(default="", env="FINNHUB_API_KEY")
    
    # Database/Cache Settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Default Trading Settings
    DEFAULT_SYMBOL: str = "BTC-USD"
    DEFAULT_PERIOD: str = "365d"
    DEFAULT_INTERVAL: str = "1d"
    DEFAULT_FORECAST_DAYS: int = 30
    
    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% máximo por posição
    STOP_LOSS_PERCENTAGE: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.15  # 15% take profit
    
    # Technical Analysis
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    MA_SHORT_PERIOD: int = 20
    MA_LONG_PERIOD: int = 50
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    
    # Output Settings
    OUTPUT_DIR: str = "outputs"
    CHART_DPI: int = 300
    CHART_FIGSIZE: tuple = (12, 8)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instância global das configurações
settings = Settings()

# Configurações de indicadores técnicos
TECHNICAL_INDICATORS = {
    'sma': {'periods': [20, 50, 200]},
    'ema': {'periods': [12, 26]},
    'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger': {'period': 20, 'std_dev': 2},
    'stochastic': {'k_period': 14, 'd_period': 3},
    'williams_r': {'period': 14},
    'atr': {'period': 14}
}

# Configurações de modelagem
MODEL_CONFIG = {
    'prophet': {
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': False,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    },
    'validation': {
        'train_size': 0.8,
        'cv_folds': 5,
        'metrics': ['mae', 'mape', 'rmse']
    }
}

# Configurações de notificação
NOTIFICATION_CONFIG = {
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',
        'sender_password': '',
        'recipients': []
    },
    'discord': {
        'enabled': False,
        'webhook_url': ''
    },
    'telegram': {
        'enabled': False,
        'bot_token': '',
        'chat_id': ''
    }
}