# EAGLE-1 v2.0 🦅

Sistema Avançado de Análise e Previsão de Criptomoedas

## ✨ Características

- 📈 **Análise Técnica Completa**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR
- 🤖 **Machine Learning**: Modelo Prophet do Facebook para previsões precisas
- 📰 **Análise de Sentimento**: Processamento de notícias com NLTK/VADER
- 🎯 **Sinais de Trading**: Geração automática de sinais de compra/venda
- 📊 **Visualizações Interativas**: Gráficos dinâmicos com Plotly
- 🌐 **Interface Web**: Dashboard moderno com Streamlit
- 📝 **Relatórios Automatizados**: Análises detalhadas em texto e CSV
- 💱 **Conversão de Moedas**: Suporte a BRL e outras moedas
- 🔧 **Sistema de Logs**: Logging profissional para debugging

## 🚀 Instalação

```bash
# 1. Clone ou baixe os arquivos do projeto
cd eagle-1

# 2. Instale as dependências
pip install -r requieriments.txt

# 3. Execute o sistema
python eagle1.py --help
```

## 📊 Uso da Interface de Linha de Comando

### Exemplo Básico
```bash
# Análise padrão do Bitcoin (30 dias)
python eagle1.py

# Ethereum com 60 dias de previsão
python eagle1.py --symbol ETH-USD --forecast_days 60

# Análise completa com gráfico interativo
python eagle1.py --symbol BTC-USD --interactive --create_sample
```

### Parâmetros Disponíveis

| Parâmetro | Descrição | Padrão |
|-----------|-----------|---------|
| `--symbol` | Criptomoeda a analisar | BTC-USD |
| `--period` | Período histórico | 365d |
| `--forecast_days` | Dias para previsão | 30 |
| `--headlines` | Arquivo de notícias | None |
| `--create_sample` | Criar exemplo de notícias | False |
| `--interactive` | Gráfico HTML interativo | False |
| `--output_dir` | Diretório de saída | outputs |
| `--no_fx` | Não buscar câmbio USD/BRL | False |

## 🌐 Interface Web

```bash
# Executar interface web (requer instalação do Streamlit)
streamlit run web_app.py
```

A interface web oferece:
- Dashboard interativo
- Análise em tempo real
- Visualizações avançadas
- Tabelas de previsões
- Sinais de trading

## 📁 Estrutura dos Arquivos de Saída

```
outputs/
├── eagle_analysis_forecast.csv    # Previsões detalhadas
├── eagle_analysis_report.txt      # Relatório textual
└── eagle_analysis_interactive.html # Gráfico interativo
```

## 📈 Indicadores Técnicos Suportados

### Tendência
- **SMA** (Simple Moving Average): 20, 50 períodos
- **EMA** (Exponential Moving Average): 12, 26 períodos
- **Bollinger Bands**: Bandas de volatilidade

### Momentum
- **RSI** (Relative Strength Index): Força relativa
- **MACD**: Convergência/divergência de médias móveis
- **Stochastic**: Oscilador estocástico
- **Williams %R**: Oscilador de Williams

### Volume
- **Volume Ratio**: Relação com média de volume
- **ATR** (Average True Range): Volatilidade média

## 🤖 Machine Learning

O sistema utiliza o **Prophet** (Facebook) que oferece:
- Detecção automática de tendências
- Sazonalidade (diária, semanal)
- Regressores externos (indicadores técnicos)
- Intervalos de confiança
- Robustez a dados faltantes

## 📰 Análise de Sentimento

Suporte a análise de notícias:
- **VADER Sentiment**: Análise de polaridade
- Formato de arquivo flexível
- Agregação temporal
- Impacto nas previsões

### Formato de Arquivo de Notícias
```
# Uma manchete por linha
Bitcoin reaches new all-time high amid institutional adoption
Major cryptocurrency exchange announces new security measures

# Ou com datas (separado por tab)
2025-09-01	Bitcoin price surges to $65,000
2025-09-02	Ethereum shows strong bullish momentum
```

## 🎯 Sinais de Trading

### Sinais de Compra (Bullish)
- RSI < 30 (oversold) + MACD bullish crossover
- Preço abaixo da Bollinger Band inferior + MACD bullish
- Combinação de indicadores de momentum

### Sinais de Venda (Bearish)
- RSI > 70 (overbought) + MACD bearish crossover  
- Preço acima da Bollinger Band superior + MACD bearish
- Indicadores de divergência

## 📊 Exemplos de Uso

### 1. Análise Rápida
```bash
python eagle1.py --symbol BTC-USD --forecast_days 7
```

### 2. Análise Completa com Sentimento
```bash
python eagle1.py --symbol ETH-USD --create_sample --interactive --forecast_days 30
```

### 3. Análise de Múltiplas Moedas
```bash
# Bitcoin
python eagle1.py --symbol BTC-USD --output_prefix btc_analysis

# Ethereum  
python eagle1.py --symbol ETH-USD --output_prefix eth_analysis

# Cardano
python eagle1.py --symbol ADA-USD --output_prefix ada_analysis
```

### 4. Análise Histórica Longa
```bash
python eagle1.py --symbol BTC-USD --period 5y --forecast_days 90 --interactive
```

## 🔧 Configurações Avançadas

### Arquivo .env (opcional)
```bash
ALPHA_VANTAGE_API_KEY=sua_chave_aqui
FINNHUB_API_KEY=sua_chave_aqui
LOG_LEVEL=INFO
OUTPUT_DIR=custom_outputs
```

### Personalização de Indicadores
Edite o arquivo `config.py` para ajustar:
- Períodos dos indicadores
- Thresholds de RSI
- Parâmetros do MACD
- Configurações do modelo Prophet

## 📝 Interpretando os Resultados

### Métricas Principais
- **Preço Atual**: Último preço conhecido
- **RSI**: < 30 = oversold, > 70 = overbought
- **MACD**: Acima da linha de sinal = bullish
- **Bollinger Bands**: Fora das bandas = movimento extremo

### Previsões
- **yhat**: Previsão central
- **yhat_lower/upper**: Intervalo de confiança (80%)
- **Variação %**: Mudança esperada em relação ao preço atual

### Sinais de Trading
- 🟢 **COMPRA**: Confluência de indicadores bullish
- 🔴 **VENDA**: Confluência de indicadores bearish  
- 🟡 **NEUTRO**: Sinais contraditórios ou ausentes

## ⚠️ Avisos Importantes

1. **Não é Aconselhamento Financeiro**: Use apenas para educação/pesquisa
2. **Mercados são Voláteis**: Criptomoedas têm alta volatilidade
3. **Backtesting Limitado**: Performance passada não garante resultados futuros
4. **Use Stop Loss**: Sempre implemente gestão de risco
5. **Diversifique**: Não invista tudo em uma única moeda

## 🐛 Solução de Problemas

### Erros Comuns

**Erro de Importação**:
```bash
pip install -r requieriments.txt
```

**Erro de Dados**:
```bash
# Verificar conectividade com Yahoo Finance
python -c "import yfinance; print(yfinance.download('BTC-USD', period='5d'))"
```

**Erro de Permissão**:
```bash
mkdir -p outputs logs
chmod 755 outputs logs
```

### Logs de Debug
```bash
# Executar com logs verbosos
python eagle1.py --verbose --symbol BTC-USD
```

Os logs são salvos em `logs/eagle1_YYYYMMDD.log`

## 🔄 Atualizações Futuras

- [ ] Suporte a mais exchanges (Binance, Coinbase Pro)
- [ ] Backtesting automatizado
- [ ] Alertas por email/Discord/Telegram
- [ ] API REST para integração
- [ ] Análise de múltiplas moedas simultânea
- [ ] Otimização de parâmetros com Grid Search
- [ ] Integração com TradingView
- [ ] Paper Trading / Simulação

## 📞 Suporte

Para questões e melhorias:
- Verifique os logs em `logs/`
- Execute com `--verbose` para mais detalhes
- Teste com dados de exemplo primeiro

## 📄 Licença

Este projeto é para fins educacionais. Use por sua conta e risco.

---

**EAGLE-1 v2.0** - Sistema Profissional de Análise de Criptomoedas 🦅