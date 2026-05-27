import numpy as np
import matplotlib.pyplot as plt
import math

def probs(sigma, T, N):
    u = np.exp(sigma*np.sqrt(T/N))
    d = 1 / u
    print(u, d)
    return u, d

def american_capped_call_binomial(S0, K, r, sigma, T, N):
    L = 0.2 * K
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
    V_tree[N] = [max(0, min(S - K, L)) for S in S_tree[N]]
    # Backward induction
    for i in reversed(range(N)):
        V_prev = []
        for j in range(i + 1):
            cont = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            exer = max(0, min(S_tree[i][j] - K, L))
            print(cont, exer)
            V_prev.append(max(cont, exer))
        V_tree[i] = V_prev
    return S_tree, V_tree

def plot_binomial_tree(S_tree, V_tree, K):
    L = 0.2 * K
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
        intrinsic = max(0, min(S_tree[i][j] - K, L))
        ax.text(x, y + 0.05, f"{val:.4f}", ha='center', va='bottom')
        ax.text(x, y - 0.05, f"{intrinsic:.4f}", ha='center', va='top')
    ax.set_axis_off()
    ax.set_title("Árbol Binomial: Valores Put Americano\n(línea superior = valor, inferior = intrínseco)")
    plt.tight_layout()
    plt.show()

# AMERICAN BINOMIAL
S0, K, r, sigma, T, N = 50, 52, 0.05, 0.25, 0.5, 3
S_tree, V_tree = american_capped_call_binomial(S0, K, r, sigma, T, N)
plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")
