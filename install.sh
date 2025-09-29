#!/bin/bash
# Script de instalação e configuração do EAGLE-1

echo "🦅 EAGLE-1 v2.0 - Script de Instalação"
echo "======================================"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale o Python 3.8 ou superior."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Criar ambiente virtual (opcional)
read -p "🤔 Deseja criar um ambiente virtual? (recomendado) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv eagle_env
    
    echo "🔧 Ativando ambiente virtual..."
    source eagle_env/bin/activate
    echo "✅ Ambiente virtual ativo"
fi

# Instalar dependências
echo "📥 Instalando dependências..."
pip install --upgrade pip

if pip install -r requieriments.txt; then
    echo "✅ Dependências instaladas com sucesso!"
else
    echo "❌ Erro ao instalar dependências. Verificando..."
    
    # Instalar uma por uma em caso de erro
    echo "🔧 Tentando instalação individual..."
    pip install pandas yfinance requests prophet nltk matplotlib numpy scikit-learn plotly
    
    # Streamlit é opcional
    read -p "🌐 Instalar Streamlit para interface web? [y/N]: " install_streamlit
    if [[ $install_streamlit =~ ^[Yy]$ ]]; then
        pip install streamlit dash python-dotenv pydantic
    fi
fi

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p outputs logs data

# Download de dados necessários para NLTK
echo "📚 Baixando dados do NLTK..."
python3 -c "import nltk; nltk.download('vader_lexicon', quiet=True); print('✅ NLTK configurado')"

# Testar instalação básica
echo "🧪 Testando instalação..."
if python3 -c "
import pandas, yfinance, requests, nltk, matplotlib
from prophet import Prophet
print('✅ Todas as dependências principais carregadas com sucesso!')
"; then
    echo "✅ Teste básico passou!"
else
    echo "⚠️  Alguns módulos podem não estar funcionando corretamente."
fi

# Executar testes automatizados
read -p "🔬 Executar testes automatizados? [y/N]: " run_tests
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "🧪 Executando testes..."
    python3 test_eagle.py
fi

# Exemplo de uso
echo ""
echo "🎉 Instalação concluída!"
echo ""
echo "📋 Próximos passos:"
echo "==================="

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "1. Ative o ambiente virtual: source eagle_env/bin/activate"
fi

echo "2. Execute uma análise básica:"
echo "   python3 eagle1.py --symbol BTC-USD --create_sample"
echo ""
echo "3. Execute a interface web (se instalou Streamlit):"
echo "   streamlit run web_app.py"
echo ""
echo "4. Para mais opções:"
echo "   python3 eagle1.py --help"
echo ""

# Informações importantes
echo "⚠️  AVISOS IMPORTANTES:"
echo "- Este é um sistema educacional, não constitui aconselhamento financeiro"
echo "- Sempre faça sua própria pesquisa antes de investir"
echo "- Use gestão de risco apropriada"
echo ""
echo "📚 Documentação completa: README.md"
echo "🐛 Para problemas: verifique os logs em logs/"

echo ""
echo "🦅 EAGLE-1 pronto para uso!"