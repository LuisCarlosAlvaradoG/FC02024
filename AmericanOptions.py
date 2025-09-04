import numpy as np
import matplotlib.pyplot as plt

#%%
def american_put_binomial(S0, K, r, T, u, d, N):
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

#%%
S0, K, r, T, u, d, N = 50, 52, 0.05, 1.0, 1.2, 0.8, 2
S_tree, V_tree = american_put_binomial(S0, K, r, T, u, d, N)
plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")
#%%
S0, K, r, T, u, d, N = 50, 52, 0.05, 1.0, 1.002, 0.998, 3000
S_tree, V_tree = american_put_binomial(S0, K, r, T, u, d, N)
#plot_binomial_tree(S_tree, V_tree, K)
print(f"El precio de la opción es {V_tree[0][0]}")