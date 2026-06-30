import numpy as np
import matplotlib.pyplot as plt
import math

#%% HELPER FUNCTIONS

def _ncdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def probs(sigma, T, N):
    """Cox-Ross-Rubinstein up and down factors."""
    u = np.exp(sigma * np.sqrt(T / N))
    d = 1.0 / u
    return u, d

#%% BJERKSUND-STENSLAND (2002) CLOSED FORM

def _phi(S, T, gamma, H, I, r, b, sigma):
    """
    Auxiliary barrier function φ(S,T,γ; H,I | r,b,σ) used in B-S.
    Computes expected present value with knock-out barrier.
    """
    if T <= 0:
        return 0.0
    
    v = sigma
    v2 = v * v
    lam = (-r + gamma*b + 0.5*gamma*(gamma-1.0)*v2) * T
    d = -(math.log(S/H) + (b + (gamma-0.5)*v2)*T) / (v*math.sqrt(T))
    k = 2.0*b/v2 + (2.0*gamma - 1.0)
    
    # Reflection coefficient
    ref = (I/S)**k
    return math.exp(lam) * (S**gamma) * (
        _ncdf(d) - ref * _ncdf(d - 2.0*math.log(I/S)/(v*math.sqrt(T)))
    )

def american_call_bjerksund(S, K, r, q, sigma, T):
    """
    Bjerksund-Stensland (2002) closed-form approximation for American call.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity (years)
    
    Returns:
    --------
    float
        American call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    
    b = r - q  # Cost of carry
    v = sigma
    v2 = v * v

    # If cost-of-carry >= r, no early exercise incentive -> return European price
    if b >= r:
        d1 = (math.log(S/K) + (b + 0.5*v2)*T) / (v*math.sqrt(T))
        d2 = d1 - v*math.sqrt(T)
        return S*math.exp((b-r)*T)*_ncdf(d1) - K*math.exp(-r*T)*_ncdf(d2)

    # Compute elasticity β
    beta = (0.5 - b/v2) + math.sqrt(((b/v2 - 0.5)**2) + 2.0*r/v2)
    
    # Asymptotic boundary B∞
    B_inf = (beta / (beta - 1.0)) * K
    
    # Initial boundary B0
    B0 = max(K, (r/b)*K) if b != 0 else K
    
    # Decay function h(T)
    h = -(b*T + 2.0*sigma*math.sqrt(T)) * (K**2 / ((B_inf - B0)*B0))
    
    # Exercise boundary X
    I = B0 + (B_inf - B0) * (1.0 - math.exp(h))

    # If current price >= boundary, exercise immediately
    if S >= I:
        return S - K

    # Coefficient α
    alpha = (I - K) * (I ** (-beta))
    
    # B-S formula with barrier functions
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
    Bjerksund-Stensland American put via duality:
    put(S,K,r,q,σ,T) = call(K,S, r'=q, q'=r, σ, T)
    """
    return american_call_bjerksund(K, S, q, r, sigma, T)

#%% BINOMIAL TREE METHODS

def american_put_binomial(S0, K, r, T, N, u=None, d=None, sigma=None):
    """
    American put via CRR binomial tree with backward induction.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity (years)
    N : int
        Number of steps
    u, d : float, optional
        Up and down factors (computed from sigma if not provided)
    sigma : float, optional
        Volatility (used to compute u,d if they're not provided)
    
    Returns:
    --------
    S_tree : list of lists
        Stock prices at each node
    V_tree : list of lists
        Option values at each node
    """
    if sigma is not None:
        u, d = probs(sigma, T, N)
    
    dt = T / N
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build stock price tree
    S_tree = [
        [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for i in range(N + 1)
    ]
    
    # Initialize option values at maturity
    V_tree = [None] * (N + 1)
    V_tree[N] = [max(K - S, 0) for S in S_tree[N]]
    
    # Backward induction: early exercise at every node
    for i in reversed(range(N)):
        V_prev = []
        for j in range(i + 1):
            continuation = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            intrinsic = max(K - S_tree[i][j], 0)
            V_prev.append(max(continuation, intrinsic))
        V_tree[i] = V_prev
    
    return S_tree, V_tree

def american_call_binomial(S0, K, r, T, N, u=None, d=None, sigma=None):
    """
    American call via CRR binomial tree with backward induction.
    Same structure as american_put_binomial but with call payoff.
    """
    if sigma is not None:
        u, d = probs(sigma, T, N)
    
    dt = T / N
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build stock price tree
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
            continuation = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            intrinsic = max(S_tree[i][j] - K, 0)
            V_prev.append(max(continuation, intrinsic))
        V_tree[i] = V_prev
    
    return S_tree, V_tree

#%% VISUALIZATION

def plot_binomial_tree(S_tree, V_tree, K, option_type="put", title_suffix=""):
    """
    Visualize the binomial tree with stock prices and option values.
    
    Each node shows:
    - Stock price (top)
    - Option value (middle, bold)
    - Intrinsic value (bottom, gray)
    
    Nodes are color-coded: red if early exercise is optimal.
    """
    N = len(S_tree) - 1
    
    # Compute positions for tree layout
    pos = {}
    for i in range(N + 1):
        for j in range(i + 1):
            x = j - i / 2  # Centered x position
            y = -i         # y increases downward
            pos[(i, j)] = (x, y)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw tree edges
    for i in range(N):
        for j in range(i + 1):
            x0, y0 = pos[(i, j)]
            x1_up, y1_up = pos[(i + 1, j + 1)]  # up move
            x1_down, y1_down = pos[(i + 1, j)]  # down move
            ax.plot([x0, x1_up], [y0, y1_up], 'k-', alpha=0.5, linewidth=1)
            ax.plot([x0, x1_down], [y0, y1_down], 'k-', alpha=0.5, linewidth=1)
    
    # Draw nodes with annotations
    for (i, j), (x, y) in pos.items():
        S = S_tree[i][j]
        V = V_tree[i][j]
        
        # Compute intrinsic value
        if option_type.lower() == "put":
            intrinsic = max(K - S, 0)
        else:  # call
            intrinsic = max(S - K, 0)
        
        # Check if early exercise is optimal (V > continuation value approximately)
        # At terminal node, V == intrinsic, so only interesting for non-terminal
        is_exercise = (i < N) and (abs(V - intrinsic) < 1e-6 or V > intrinsic + 1e-4)
        
        # Node color: red if exercise, light blue otherwise
        node_color = 'red' if is_exercise else 'lightblue'
        
        # Draw circle for node
        circle = plt.Circle((x, y), 0.25, color=node_color, ec='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        
        # Annotate: stock price, option value, intrinsic
        ax.text(x, y + 0.06, f"S={S:.0f}", ha='center', va='center', 
                fontsize=8, weight='bold', zorder=11)
        ax.text(x, y - 0.01, f"{V:.4f}", ha='center', va='center', 
                fontsize=9, weight='bold', zorder=11)
        ax.text(x, y - 0.08, f"int={intrinsic:.1f}", ha='center', va='center', 
                fontsize=7, color='gray', zorder=11)
    
    # Clean up axes
    ax.set_aspect('equal')
    ax.set_xlim(-N/2 - 1, N/2 + 1)
    ax.set_ylim(-N - 0.5, 0.5)
    ax.axis('off')
    
    # Title and legend
    option_label = "American Put" if option_type.lower() == "put" else "American Call"
    ax.text(0.5, 1.05, f"{option_label} - CRR Binomial Tree {title_suffix}", 
            ha='center', va='bottom', transform=ax.transAxes, fontsize=12, weight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Wait (continuation > intrinsic)'),
        Patch(facecolor='red', edgecolor='black', label='Exercise early (intrinsic > continuation)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=9)
    
    plt.tight_layout()
    return fig

#%% MAIN EXAMPLES FROM THE PRESENTATION

S0, K, r, T, u, d, N, sigma = 50, 52, 0.05, 1.0, 1.2, 0.8, 10, 0.25784161

print(f"\nParameters:")
print(f"  S₀ = ${S0},  K = ${K},  r = {r*100}%,  T = {T} year")
print(f"  u = {u},  d = {d},  N = {N} steps")
print(f"  dt = {T/N}, p = (e^(r·dt) - d)/(u - d)")

S_tree, V_tree = american_put_binomial(S0, K, r, T, N, sigma = sigma)

print(f"\nStock Price Tree:")
for i, row in enumerate(S_tree):
    print(f"  t={i}: {[f'${s:.0f}' for s in row]}")

print(f"\nOption Value Tree:")
for i, row in enumerate(V_tree):
    print(f"  t={i}: {[f'{v:.4f}' for v in row]}")

print(f"\n>>> American Put Price = ${V_tree[0][0]:.4f}")

# Visualize
fig1 = plot_binomial_tree(S_tree, V_tree, K, option_type="put", 
                          title_suffix="(N=2, S₀=50, K=52, r=5%)")
plt.show()

#%% EXAMPLE 2: Convergence with Large N

print("EXAMPLE 2: Convergence to Black-Scholes as N increases")

S0, K, r, T, sigma = 50, 52, 0.05, 1.0, 0.25784161  # σ ≈ 25.78%

print(f"\nParameters:")
print(f"  S₀ = ${S0},  K = ${K},  r = {r*100}%,  T = {T} year")
print(f"  σ = {sigma*100:.2f}%")

print(f"\n{'Steps (N)':<12} {'American Put':<15} {'Convergence':<15}")
print("-" * 42)

for N in [10, 50, 100, 500, 1000]:
    S_tree, V_tree = american_put_binomial(S0, K, r, T, N, sigma=sigma)
    price = V_tree[0][0]
    print(f"{N:<12} ${price:<14.6f}")

#%% Bjerksund-Stensland comparison
bs_price = american_put_bjerksund(S0, K, r, 0.0, sigma, T)
print(f"{'B-S 2002':<12} ${bs_price:<14.6f}")

