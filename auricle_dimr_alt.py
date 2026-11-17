"""SLS (RMSprop + RLS) y PCA-SVD.  H = tanh(X V); reconstrucción H U.
U: mínimos cuadrados regularizados (equivale al RLS recursivo con olvido 1 y P0 = I/ridge).
V: RMSprop sobre el costo de reconstrucción (U se trata como constante en cada paso: optimización alternada).
"""
import numpy as np


def entrenar_sls(X, J=20, epochs=500, lr=0.001, beta=0.9, ridge=0.001, seed=42,
                 eps=1e-8, clip=None, lr_final=None, tol=None, paciencia=30):
    """Parámetros opcionales (por defecto reproducen el Experimento 1):
    clip      norma máxima del gradiente (None = sin recorte)
    lr_final  si se indica, la tasa decae con coseno de lr a lr_final (suprime los picos)
    tol       parada temprana si el mejor costo no mejora > tol (relativo) en `paciencia` épocas
    Devuelve el V de MENOR costo (no el de la última época) y su U."""
    X = np.asarray(X, float)
    if not 1 <= J < X.shape[1] or ridge <= 0:
        raise ValueError("J o regularización inválidos")
    V = np.random.default_rng(seed).normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], J))
    acc = np.zeros_like(V)
    I = np.eye(J)

    def evaluar(V):
        H = np.tanh(X @ V)
        U = np.linalg.solve(H.T @ H + ridge * I, H.T @ X)
        E = H @ U - X
        costo = (np.sum(E ** 2) + ridge * np.sum(U ** 2)) / (2 * len(X))
        return H, U, E, costo

    costos, mejor, V_mejor, sin_mejora = [], np.inf, V.copy(), 0
    for t in range(epochs):
        H, U, E, c = evaluar(V)
        costos.append(c)
        if c < mejor * (1 - (tol or 0)):
            sin_mejora = 0
        else:
            sin_mejora += 1
        if c < mejor:
            mejor, V_mejor = c, V.copy()
        if tol is not None and sin_mejora >= paciencia:
            break
        g = X.T @ ((E @ U.T) * (1 - H ** 2)) / len(X)
        if clip is not None:
            n = np.linalg.norm(g)
            if n > clip:
                g *= clip / n
        lr_t = lr if lr_final is None else lr_final + (lr - lr_final) * 0.5 * (1 + np.cos(np.pi * t / epochs))
        acc = beta * acc + (1 - beta) * g ** 2
        V -= lr_t * g / (np.sqrt(acc) + eps)
    _, U, _, c = evaluar(V)
    if c < mejor:
        V_mejor = V
    _, U, _, _ = evaluar(V_mejor)
    return V_mejor, U, np.asarray(costos)


def pca_svd(H, objetivo=0.99, k=None):
    """Centra H y devuelve (media, componentes, varianza acumulada).
    Si se da k se fija el nº de componentes; si no, el mínimo que alcanza `objetivo`."""
    mu = H.mean(axis=0)
    _, s, vt = np.linalg.svd(H - mu, full_matrices=False)
    if np.sum(s ** 2) == 0:
        raise ValueError("Sin variabilidad latente")
    acumulada = np.cumsum(s ** 2) / np.sum(s ** 2)
    if k is None:
        k = int(np.searchsorted(acumulada, objetivo) + 1)
    k = int(min(len(s), max(1, k)))
    return mu, vt[:k].T, acumulada


def transformar(X, modelo):
    X = np.asarray(X, float)
    H = np.tanh(X @ modelo['V']) if modelo['V'].size else X
    return (H - modelo['media_Xr']) @ modelo['Vpca']


def error_reconstruccion(X, modelo):
    """Error relativo ||H U - X||² / ||X||² (sirve también en Test para ver si el SLS generaliza)."""
    if not modelo['V'].size:
        return float('nan')
    X = np.asarray(X, float)
    E = np.tanh(X @ modelo['V']) @ modelo['U'] - X
    return float(np.sum(E ** 2) / np.sum(X ** 2))


def ajustar(X, J=20, objetivo=0.99, sls=True, pca=True, k=None, **kwargs):
    """sls/pca permiten las 4 arquitecturas del estudio de ablación."""
    X = np.asarray(X, float)
    if sls:
        V, U, costos = entrenar_sls(X, J=J, **kwargs)
        H = np.tanh(X @ V)
    else:
        V, U, costos, H = np.zeros((0, 0)), np.zeros((0, 0)), np.array([]), X
    if pca:
        mu, P, acum = pca_svd(H, objetivo, k)
    else:
        mu, P, acum = np.zeros(H.shape[1]), np.eye(H.shape[1]), np.array([1.0])
    return dict(V=V, U=U, media_Xr=mu, Vpca=P), costos, acum
