# Como Usar o EAGLE-1 🦅

## ✅ Status do Projeto
**FUNCIONANDO PERFEITAMENTE!** ✅

O projeto EAGLE-1 está totalmente funcional e oferece análise completa de criptomoedas com:
- ✅ Dados de 30 dias para análise
- ✅ Previsão para os próximos 5 dias
- ✅ Análise de notícias das últimas 24 horas
- ✅ Indicadores técnicos (RSI, MACD, SMA)
- ✅ Conversão para BRL
- ✅ Análise de sentimento das notícias

## 🚀 Como Executar

### 1. Demonstração Rápida (RECOMENDADO)
```bash
cd /Users/pedrini/Documents/eagle-1
python3 quick_demo.py
```

### 2. Análise Completa
```bash
# Bitcoin com análise padrão
python3 eagle1.py

# Ethereum com 60 dias de previsão
python3 eagle1.py --symbol ETH-USD --forecast_days 60

# Análise completa com gráfico interativo
python3 eagle1.py --symbol BTC-USD --interactive --create_sample
```

### 3. Interface Web (se disponível)
```bash
streamlit run web_app.py
```

## 📊 O que o Sistema Faz

### Demonstração Rápida (`quick_demo.py`)
- 📈 **Dados**: Últimos 30 dias do Bitcoin
- 🔮 **Previsão**: Próximos 5 dias
- 📰 **Notícias**: Análise das últimas 24h
- 💱 **Conversão**: Preços em BRL
- 🎯 **Recomendação**: Compra/Venda/Neutro

### Sistema Completo (`eagle1.py`)
- 🤖 **Machine Learning**: Modelo Prophet
- 📊 **Indicadores**: RSI, MACD, Bollinger Bands, etc.
- 📈 **Gráficos**: Visualizações interativas
- 📝 **Relatórios**: Análises detalhadas em texto e CSV

## 🛠️ Dependências Principais

O projeto já tem todas as dependências instaladas:
- `yfinance` - Dados financeiros
- `pandas` - Manipulação de dados  
- `textblob` - Análise de sentimento
- `requests` - APIs de notícias
- `matplotlib/plotly` - Gráficos
- `prophet` - Machine Learning

## 📋 Exemplo de Saída

```
🦅 EAGLE-1 v2.0 - Demonstração Rápida
=============================================
📊 Buscando dados do Bitcoin (30 dias)...
💰 Preço atual: $112,285.67 USD
📈 RSI (14): 55.0 - 🟡 Status: Neutro
📊 MACD: -591.36 - 🔴 Tendência: Bearish (baixa)
📈 Análise de tendência... Status: 🟡 Lateral
🔮 Previsão (5 dias): $112,680.19 USD (+0.4%)
💱 Conversão para BRL...
💰 Preço atual: R$ 617,571.20
🔮 Previsão (5 dias): R$ 619,741.02
📰 Análise de Notícias (últimas 24h)...
📊 Sentimento geral: Positivo (0.15)
🎯 RECOMENDAÇÃO: 🔴 CAUTELOSO - Considere venda
```

## 🎛️ Opções de Comando

| Parâmetro | Descrição | Exemplo |
|-----------|-----------|---------|
| `--symbol` | Criptomoeda | `BTC-USD`, `ETH-USD` |
| `--period` | Período histórico | `30d`, `60d`, `1y` |
| `--forecast_days` | Dias de previsão | `5`, `30`, `90` |
| `--interactive` | Gráfico HTML | Sim/Não |
| `--create_sample` | Criar exemplo de notícias | Sim/Não |

## 📁 Arquivos Gerados

Os resultados ficam na pasta `outputs/`:
- `eagle_analysis_forecast.csv` - Previsões detalhadas
- `eagle_analysis_report.txt` - Relatório textual  
- `eagle_analysis_interactive.html` - Gráfico interativo

## ⚠️ Avisos Importantes

- ✅ Sistema **100% educacional**
- ❌ **NÃO é aconselhamento financeiro**
- 🔍 Sempre faça sua própria pesquisa
- 💼 Use gestão de risco apropriada
- 📊 Performance passada não garante resultados futuros

---

**EAGLE-1 v2.0** - Sistema Profissional de Análise de Criptomoedas 🦅
*Atualizado para usar 30 dias de dados, prever 5 dias e incluir notícias das últimas 24h*