import numpy as np
import matplotlib.pyplot as plt
import math

#%%
def probs(sigma, T, N):
    u = np.exp(sigma*np.sqrt(T/N))
    d = 1 / u
    return u, d

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# Función auxiliar φ(s,t,γ; H,I | r,b,σ) usada en BS1993/BS2002
# (forma tomada de implementaciones académicas habituales)
def _phi(S, T, gamma, H, I, r, b, sigma):
    if T <= 0:
        return 0.0
    v = sigma
    v2 = v*v
    lam = (-r + gamma*b + 0.5*gamma*(gamma-1.0)*v2) * T
    d  = -(math.log(S/H) + (b + (gamma-0.5)*v2)*T) / (v*math.sqrt(T))
    k  = 2.0*b/v2 + (2.0*gamma - 1.0)
    # término de reflexión
    ref = (I/S)**k
    return math.exp(lam) * (S**gamma) * (_ncdf(d) - ref * _ncdf(d - 2.0*math.log(I/S)/(v*math.sqrt(T))))

def american_call_bjerksund(S, K, r, q, sigma, T):
    """
    Bjerksund-Stensland (2002) para CALL americano con dividendos continuos q.
    Si no hay ejercicio temprano (p.ej. b>=r), retorna el call europeo.
    """
    if T <= 0:
        return max(S - K, 0.0)
    b = r - q
    v = sigma
    v2 = v*v

    # Si el costo-de-carry >= r, el ejercicio temprano no conviene -> europeo
    if b >= r:
        d1 = (math.log(S/K) + (b + 0.5*v2)*T) / (v*math.sqrt(T))
        d2 = d1 - v*math.sqrt(T)
        return S*math.exp((b-r)*T)*_ncdf(d1) - K*math.exp(-r*T)*_ncdf(d2)

    # Parámetros clave de BS2002
    beta = (0.5 - b/v2) + math.sqrt(((b/v2 - 0.5)**2) + 2.0*r/v2)
    B_inf = (beta/(beta-1.0)) * K
    B0 = max(K, (r/b)*K)  # frontera inicial
    h = -(b*T + 2.0*sigma*math.sqrt(T)) * (B0/(B_inf - B0))
    I = B0 + (B_inf - B0) * (1.0 - math.exp(h))  # frontera plana “óptima” BS

    # Si estamos por encima de la frontera, el valor es el payoff intrínseco
    if S >= I:
        return S - K

    # Constante α y fórmula cerrada con funciones φ
    alpha = (I - K) * (I ** (-beta))
    term = (
        alpha * (S ** beta)
        - alpha * _phi(S, T, beta, I, I, r, b, sigma)
        + _phi(S, T, 1.0, I, I, r, b, sigma)
        - _phi(S, T, 1.0, K, I, r, b, sigma)
        - K * _phi(S, T, 0.0, I, I, r, b, sigma)
        + K * _phi(S, T, 0.0, K, I, r, b, sigma)
    )
    return term

def american_put_bjerksund(S, K, r, q, sigma, T):
    """
    Put americano vía dualidad de Bjerksund–Stensland:
    p(S,K,r,q,σ,T) = c(K,S, r’=q, q’=r, σ, T)
    """
    return american_call_bjerksund(K, S, q, r, sigma, T)

#%%
def american_put_binomial(S0, K, r, T, N, u = None, d = None, sigma = None):
    if sigma:
        u, d = probs(sigma, T, N)
    dt = T / N
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    # Build price tree
    S_tree = [
        [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for i in range(N + 1)
    ]
    # Initialize option values at maturity
    V_tree = [None] * (N + 1)
    V_tree[N] = [max(K - S, 0) for S in S_tree[N]]
    # Backward induction
    for i in reversed(range(N)):
        V_prev = []
        for j in range(i + 1):
            cont = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            exer = max(K - S_tree[i][j], 0)
            V_prev.append(max(cont, exer))
        V_tree[i] = V_prev
    return S_tree, V_tree
#%%
def american_call_binomial(S0, K, r, T, N, u = None, d = None, sigma = None):
    if sigma:
        u, d = probs(sigma, T, N)
    dt = T / N
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    # Build price tree
    S_tree = [
        [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for i in range(N + 1)
    ]
    # Initialize option values at maturity
    V_tree = [None] * (N + 1)
    V_tree[N] = [max(S - K, 0) for S in S_tree[N]]
    # Backward induction
    for i in reversed(range(N)):
        V_prev = []
        for j in range(i + 1):
            cont = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            exer = max(S_tree[i][j] - K, 0)
            V_prev.append(max(cont, exer))
        V_tree[i] = V_prev
    return S_tree, V_tree
#%%
def plot_binomial_tree(S_tree, V_tree, K):
    N = len(S_tree) - 1
    pos = {}
    for i in range(N + 1):
        for j in range(i + 1):
            x = j - i/2
            y = -i
            pos[(i, j)] = (x, y)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Draw edges
    for i in range(N):
        for j in range(i + 1):
            x0, y0 = pos[(i, j)]
            x1, y1 = pos[(i+1, j)]
            x2, y2 = pos[(i+1, j+1)]
            ax.plot([x0, x1], [y0, y1], linestyle='-')
            ax.plot([x0, x2], [y0, y2], linestyle='-')
    # Annotate nodes
    for (i, j), (x, y) in pos.items():
        # Option value (continuation or immediate optimal)
        val = V_tree[i][j]
        # Intrinsic value
        intrinsic = max(K - S_tree[i][j], 0)
        ax.text(x, y + 0.05, f"{val:.4f}", ha='center', va='bottom')
        ax.text(x, y - 0.05, f"{intrinsic:.4f}", ha='center', va='top')
    ax.set_axis_off()
    ax.set_title("Árbol Binomial: Valores Put Americano\n(línea superior = valor, inferior = intrínseco)")
    plt.tight_layout()
    plt.show()

#%% AMERICAN BINOMIAL
S0, K, r, T, u, d, N = 50, 52, 0.05, 1.0, 1.2, 0.8, 2
S_tree, V_tree = american_put_binomial(S0, K, r, T, N, u, d)
plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")
#%% AMERICAN BINOMIAL SIGMA
S0, K, r, T, N, sigma = 50, 52, 0.05, 1.0, 2, 0.25784161
S_tree, V_tree = american_put_binomial(S0, K, r, T, N, sigma = sigma)
#plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")
#%% AMERICAN BINOMIAL INF
S0, K, r, T, N, sigma = 50, 52, 0.05, 1.0, 10000, 0.25784161
S_tree, V_tree = american_put_binomial(S0, K, r, T, N, sigma = sigma)
#plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")

#%% BJERKSUND
S0, K, r, q, T = 50.0, 52.0, 0.05, 0.0, 1.0
sigma = math.log(1.2)/np.sqrt(.5)   # ≈ 18.2321% anual
put_bs2002  = american_put_bjerksund(S0, K, r, q, sigma, T)
call_bs2002 = american_call_bjerksund(S0, K, r, q, sigma, T)
print(put_bs2002, call_bs2002)


#%% EJERCICIO CLASE
S0, K, r, T, N, sigma = 50, 52, 0.05, 1.0, 4, 0.15
S_tree, V_tree = american_call_binomial(S0, K, r, T, N, sigma = sigma)
#plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")