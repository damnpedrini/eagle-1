import numpy as np
import math
from scipy.stats import norm, gamma
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def moving_average(series, window=7):
    if len(series) < window:
        raise ValueError("Not Enaught data to calculate moving average.")
    return sum(series[-window:]) / window

def calculate_volatility(prices, window=30):
    """Calcula a volatilidade histórica dos preços"""
    if len(prices) < 2:
        return 0.2  # volatilidade padrão de 20%
    
    returns = []
    for i in range(1, len(prices)):
        returns.append(np.log(prices[i] / prices[i-1]))
    
    volatility = np.std(returns) * np.sqrt(252)  # Anualizada
    return volatility

def black_scholes_forecast(current_price, volatility, risk_free_rate=0.05, time_horizon=1):
    """
    Aplica o modelo Black-Scholes para previsão de preços
    current_price: preço atual
    volatility: volatilidade histórica
    risk_free_rate: taxa livre de risco (5% ao ano por padrão)
    time_horizon: horizonte de tempo em anos (1/365 para 1 dia)
    """
    # Movimento browniano geométrico (base do Black-Scholes)
    drift = risk_free_rate - 0.5 * volatility**2
    random_shock = np.random.normal(0, 1)
    
    # Preço futuro usando a fórmula do Black-Scholes
    future_price = current_price * np.exp(drift * time_horizon + volatility * math.sqrt(time_horizon) * random_shock)
    
    return future_price

def black_scholes_multi_day_forecast(current_price, volatility, days=7, risk_free_rate=0.05):
    """
    Gera previsão para múltiplos dias usando Black-Scholes
    """
    forecasts = []
    price = current_price
    
    for day in range(days):
        time_horizon = 1/365  # 1 dia em anos
        price = black_scholes_forecast(price, volatility, risk_free_rate, time_horizon)
        forecasts.append(price)
    
    return forecasts

# ========== PROCESSO DE WIENER + MOVIMENTO BROWNIANO GEOMÉTRICO ==========
def geometric_brownian_motion(S0, mu, sigma, T, N, paths=1000):
    """
    Movimento Browniano Geométrico com múltiplos caminhos
    S0: preço inicial
    mu: drift (retorno esperado)
    sigma: volatilidade
    T: tempo total
    N: número de passos
    paths: número de simulações
    """
    dt = T / N
    prices = np.zeros((paths, N + 1))
    prices[:, 0] = S0
    
    for i in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        prices[:, i] = prices[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return prices

# ========== MONTE CARLO SIMULATION ==========
def monte_carlo_forecast(current_price, volatility, drift, days=7, simulations=10000):
    """
    Simulação Monte Carlo para previsão de preços
    """
    dt = 1/365  # 1 dia
    final_prices = []
    
    for _ in range(simulations):
        price = current_price
        for day in range(days):
            random_shock = np.random.normal(0, 1)
            price = price * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shock)
        final_prices.append(price)
    
    # Estatísticas da simulação
    mean_price = np.mean(final_prices)
    std_price = np.std(final_prices)
    percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
    
    return {
        'mean': mean_price,
        'std': std_price,
        'percentiles': percentiles,
        'all_prices': final_prices
    }

# ========== MODELO DE HESTON ==========
class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma_v, rho, r):
        self.S0 = S0      # preço inicial
        self.v0 = v0      # volatilidade inicial
        self.kappa = kappa  # velocidade de reversão à média
        self.theta = theta  # volatilidade de longo prazo
        self.sigma_v = sigma_v  # vol da vol
        self.rho = rho    # correlação
        self.r = r        # taxa livre de risco
    
    def simulate_paths(self, T, N, paths=1000):
        dt = T / N
        S = np.zeros((paths, N + 1))
        v = np.zeros((paths, N + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(1, N + 1):
            # Gerando choques correlacionados
            Z1 = np.random.standard_normal(paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(paths)
            
            # Processo da volatilidade (CIR)
            v[:, i] = np.abs(v[:, i-1] + self.kappa * (self.theta - v[:, i-1]) * dt + 
                           self.sigma_v * np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z2)
            
            # Processo do preço
            S[:, i] = S[:, i-1] * np.exp((self.r - 0.5 * v[:, i-1]) * dt + 
                                       np.sqrt(v[:, i-1]) * np.sqrt(dt) * Z1)
        
        return S, v

# ========== SABR MODEL ==========
class SABRModel:
    def __init__(self, alpha, beta, rho, nu):
        self.alpha = alpha  # volatilidade inicial
        self.beta = beta    # elasticidade
        self.rho = rho      # correlação
        self.nu = nu        # vol da vol
    
    def implied_volatility(self, F, K, T):
        """Calcula volatilidade implícita SABR"""
        if F == K:
            # At-the-money
            return self.alpha / (F**(1 - self.beta)) * (
                1 + ((1 - self.beta)**2 / 24 * self.alpha**2 / F**(2 - 2*self.beta) + 
                     0.25 * self.rho * self.beta * self.nu * self.alpha / F**(1 - self.beta) + 
                     (2 - 3*self.rho**2) / 24 * self.nu**2) * T
            )
        else:
            # Out-of-the-money approximation
            z = self.nu / self.alpha * (F * K)**((1 - self.beta)/2) * np.log(F/K)
            x_z = np.log((np.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))
            
            return self.alpha / ((F*K)**((1-self.beta)/2) * (1 + (1-self.beta)**2/24 * np.log(F/K)**2)) * z/x_z

# ========== BINOMIAL/TRINOMIAL TREES ==========
def binomial_tree_forecast(S0, r, sigma, T, N):
    """
    Árvore binomial para previsão de preços
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # movimento para cima
    d = 1 / u                        # movimento para baixo
    p = (np.exp(r * dt) - d) / (u - d)  # probabilidade risk-neutral
    
    # Construir árvore
    tree = np.zeros((N + 1, N + 1))
    
    # Preços finais
    for j in range(N + 1):
        tree[N, j] = S0 * (u ** (N - j)) * (d ** j)
    
    return tree[N, :]

def trinomial_tree_forecast(S0, r, sigma, T, N):
    """
    Árvore trinomial para previsão de preços
    """
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    
    # Probabilidades
    pu = 1/6 + (r - 0.5 * sigma**2) * np.sqrt(dt / (12 * sigma**2))
    pm = 2/3
    pd = 1/6 - (r - 0.5 * sigma**2) * np.sqrt(dt / (12 * sigma**2))
    
    # Construir árvore (simplificado)
    prices = []
    for i in range(-N, N + 1):
        price = S0 * np.exp(i * dx)
        prices.append(price)
    
    return np.array(prices)

# ========== JUMP DIFFUSION - MERTON ==========
def merton_jump_diffusion(S0, mu, sigma, lam, mu_j, sigma_j, T, N, paths=1000):
    """
    Modelo de Jump Diffusion de Merton
    lam: intensidade dos saltos
    mu_j: média dos saltos
    sigma_j: volatilidade dos saltos
    """
    dt = T / N
    prices = np.zeros((paths, N + 1))
    prices[:, 0] = S0
    
    for i in range(1, N + 1):
        # Componente browniano
        Z = np.random.standard_normal(paths)
        brownian = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Componente de salto (Poisson)
        N_jumps = np.random.poisson(lam * dt, paths)
        jump_component = np.zeros(paths)
        
        for j, n_jump in enumerate(N_jumps):
            if n_jump > 0:
                jumps = np.random.normal(mu_j, sigma_j, n_jump)
                jump_component[j] = np.sum(jumps)
        
        prices[:, i] = prices[:, i-1] * np.exp(brownian + jump_component)
    
    return prices

# ========== VARIANCE GAMMA MODEL ==========
def variance_gamma_model(S0, mu, sigma, nu, T, N, paths=1000):
    """
    Modelo Variance Gamma
    nu: parâmetro de forma
    """
    dt = T / N
    prices = np.zeros((paths, N + 1))
    prices[:, 0] = S0
    
    for i in range(1, N + 1):
        # Tempo aleatório Gamma
        gamma_time = np.random.gamma(dt/nu, nu, paths)
        
        # Movimento browniano subordinado
        Z = np.random.normal(0, 1, paths)
        log_return = mu * gamma_time + sigma * np.sqrt(gamma_time) * Z
        
        prices[:, i] = prices[:, i-1] * np.exp(log_return)
    
    return prices

# ========== MODELOS GARCH ==========
class GARCHModel:
    def __init__(self, returns):
        self.returns = returns
        self.n = len(returns)
    
    def garch_11_log_likelihood(self, params):
        """Log-likelihood para GARCH(1,1)"""
        omega, alpha, beta = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return -np.inf
        
        sigma2 = np.zeros(self.n)
        sigma2[0] = np.var(self.returns)
        
        log_likelihood = 0
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * sigma2[t-1]
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + self.returns[t]**2 / sigma2[t])
        
        return -log_likelihood
    
    def fit_garch_11(self):
        """Estima parâmetros GARCH(1,1)"""
        initial_guess = [0.01, 0.1, 0.8]
        bounds = [(1e-6, None), (0, 1), (0, 1)]
        
        result = minimize(self.garch_11_log_likelihood, initial_guess, bounds=bounds)
        return result.x if result.success else initial_guess
    
    def forecast_volatility(self, params_garch, horizon=7):
        """Previsão de volatilidade usando GARCH"""
        omega, alpha, beta = params_garch
        
        # Última variância
        last_return = self.returns[-1]
        last_variance = omega / (1 - alpha - beta)  # Variância incondicional
        
        forecasts = []
        current_variance = omega + alpha * last_return**2 + beta * last_variance
        
        for h in range(1, horizon + 1):
            if h == 1:
                forecast_var = current_variance
            else:
                # Variância incondicional para horizontes longos
                forecast_var = omega + (alpha + beta) * forecast_var
            
            forecasts.append(np.sqrt(forecast_var))
        
        return forecasts

# ========== PDEs PARA OPÇÕES (BLACK-SCHOLES PDE) ==========
def black_scholes_pde_solution(S, K, T, r, sigma, option_type='call'):
    """
    Solução analítica da PDE de Black-Scholes
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

# ========== FUNÇÃO PRINCIPAL PARA USAR TODOS OS MODELOS ==========
def advanced_forecast_suite(prices, current_price, days=7):
    """
    Executa todos os modelos avançados e retorna resultados consolidados
    """
    results = {}
    
    # Calcular retornos
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(252)
    drift = np.mean(returns) * 252
    
    # 1. Monte Carlo
    results['monte_carlo'] = monte_carlo_forecast(current_price, volatility, drift, days)
    
    # 2. Movimento Browniano Geométrico
    gbm_paths = geometric_brownian_motion(current_price, drift, volatility, days/365, days, 1000)
    results['gbm_final_prices'] = gbm_paths[:, -1]
    results['gbm_mean'] = np.mean(gbm_paths[:, -1])
    
    # 3. Modelo de Heston
    heston = HestonModel(current_price, volatility**2, 2.0, volatility**2, 0.3, -0.5, 0.05)
    heston_S, heston_v = heston.simulate_paths(days/365, days, 1000)
    results['heston_final_prices'] = heston_S[:, -1]
    results['heston_mean'] = np.mean(heston_S[:, -1])
    
    # 4. Jump Diffusion
    jump_paths = merton_jump_diffusion(current_price, drift, volatility, 0.1, 0, 0.2, days/365, days, 1000)
    results['jump_final_prices'] = jump_paths[:, -1]
    results['jump_mean'] = np.mean(jump_paths[:, -1])
    
    # 5. Variance Gamma
    vg_paths = variance_gamma_model(current_price, drift, volatility, 0.2, days/365, days, 1000)
    results['vg_final_prices'] = vg_paths[:, -1]
    results['vg_mean'] = np.mean(vg_paths[:, -1])
    
    # 6. GARCH
    try:
        garch_model = GARCHModel(returns)
        garch_params = garch_model.fit_garch_11()
        volatility_forecast = garch_model.forecast_volatility(garch_params, days)
        results['garch_volatility'] = volatility_forecast
    except:
        results['garch_volatility'] = [volatility] * days
    
    # 7. Árvores
    binomial_prices = binomial_tree_forecast(current_price, 0.05, volatility, days/365, days)
    results['binomial_prices'] = binomial_prices
    results['binomial_mean'] = np.mean(binomial_prices)
    
    # 8. Trinomial Tree
    trinomial_prices = trinomial_tree_forecast(current_price, 0.05, volatility, days/365, days)
    results['trinomial_prices'] = trinomial_prices
    results['trinomial_mean'] = np.mean(trinomial_prices)
    
    # 9. SABR Model (Volatilidade Implícita)
    try:
        sabr = SABRModel(alpha=volatility, beta=0.5, rho=-0.3, nu=0.4)
        sabr_iv = sabr.implied_volatility(current_price, current_price, days/365)
        results['sabr_iv'] = sabr_iv
    except:
        results['sabr_iv'] = volatility
    
    # 10. Black-Scholes PDE
    bs_call = black_scholes_pde_solution(current_price, current_price, days/365, 0.05, volatility, 'call')
    results['bs_call_price'] = bs_call
    
    return results