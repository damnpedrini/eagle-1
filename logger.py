"""
Sistema de logging profissional para EAGLE-1
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

class EagleLogger:
    """Sistema de logging customizado para EAGLE-1"""
    
    def __init__(self, name: str = "EAGLE-1", log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Evitar duplicação de handlers
        if logger.handlers:
            return logger
        
        # Formato personalizado
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler para arquivo
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"eagle1_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

# Instância global do logger
eagle_logger = EagleLogger()

# Funções de conveniência
def log_info(message: str, **kwargs):
    eagle_logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    eagle_logger.error(message, **kwargs)

def log_warning(message: str, **kwargs):
    eagle_logger.warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    eagle_logger.debug(message, **kwargs)

def log_critical(message: str, **kwargs):
    eagle_logger.critical(message, **kwargs)

# Decorator para logging de funções
def log_function_calls(func):
    """Decorator para logar chamadas de função"""
    def wrapper(*args, **kwargs):
        eagle_logger.debug(f"Chamando {func.__name__} com args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            eagle_logger.debug(f"{func.__name__} executada com sucesso")
            return result
        except Exception as e:
            eagle_logger.error(f"Erro em {func.__name__}: {str(e)}")
            raise
    return wrapper