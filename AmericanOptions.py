import numpy as np
import matplotlib.pyplot as plt

def american_put_binomial(S0, K, r, T, u, d, N):
    """
    Construye el árbol binomial y calcula el valor de un put americano.
    Devuelve dos listas de listas:
      - S_tree[i] = precios del subyacente en el paso i (i = 0..N)
      - V_tree[i] = valores del put en cada nodo del paso i
    """
    dt = T / N
    disc = np.exp(-r * dt)                  # factor de descuento e^{-r dt}
    p = (np.exp(r * dt) - d) / (u - d)      # probabilidad neutral al riesgo

    # 1) Generar árbol de precios del subyacente
    S_tree = [
        [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for i in range(N + 1)
    ]

    # 2) Valor en el vencimiento (intrínseco)
    V_tree = [None] * (N + 1)
    V_tree[N] = [max(K - S, 0) for S in S_tree[N]]

    # 3) Backward induction con posibilidad de ejercicio
    for i in reversed(range(N)):
        V_prev = []
        for j in range(i + 1):
            cont = disc * (p * V_tree[i + 1][j + 1] + (1 - p) * V_tree[i + 1][j])
            exer = max(K - S_tree[i][j], 0)
            V_prev.append(max(cont, exer))
        V_tree[i] = V_prev

    return S_tree, V_tree

def plot_tree_profiles(S_tree, V_tree, title):
    """
    Dibuja un perfil (líneas) del valor de la opción vs precio subyacente
    para cada paso t = 0..N.
    """
    N = len(V_tree) - 1
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis', N + 1)
    for i in range(N + 1):
        plt.plot(
            S_tree[i], 
            V_tree[i], 
            color=cmap(i), 
            label=f't = {i}'
        )
    plt.xlabel('Precio subyacente $S$')
    plt.ylabel('Valor del put americano')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()


# Parámetros comunes
S0 = 50
K  = 52
r  = 0.05
u  = 1.2
d  = 0.8

# =========================
# Ejemplo 1: N grande (T = 1, N = 50)
# =========================
T1, N1 = 1.0, 50
S1, V1 = american_put_binomial(S0, K, r, T1, u, d, N1)
print(f'Valor aproximado put americano (T={T1}, N={N1}): {V1[0][0]:.4f}')
plot_tree_profiles(
    S1, V1, 
    title=f'Perfil de valores del put (T={T1}, N={N1})'
)

# =========================
# Ejemplo 2: T más amplio (T = 2, N = 100)
# =========================
T2, N2 = 2.0, 100
S2, V2 = american_put_binomial(S0, K, r, T2, u, d, N2)
print(f'Valor put americano (T={T2}, N={N2}): {V2[0][0]:.4f}')
plot_tree_profiles(
    S2, V2, 
    title=f'Perfil de valores del put (T={T2}, N={N2})'
)
