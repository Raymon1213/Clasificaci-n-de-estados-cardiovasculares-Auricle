import numpy as np

def entrenar_sls(X, J=20, epochs=500, lr=0.001, beta=0.9, ridge=0.001, seed=42):
    X = np.asarray(X, float)
    if not 1 <= J < X.shape[1] or ridge <= 0:
        raise ValueError("J o regularización inválidos")
    V = np.random.default_rng(seed).normal(0, 1/np.sqrt(X.shape[1]), (X.shape[1], J))
    acc = np.zeros_like(V)
    costos = []
    def rls(H):
        return np.linalg.solve(H.T @ H + ridge*np.eye(J), H.T @ X)
    for _ in range(epochs):
        H = np.tanh(X @ V)
        U = rls(H)
        E = H @ U - X
        costos.append((np.sum(E**2) + ridge*np.sum(U**2))/(2*len(X)))
        g = X.T @ ((E @ U.T)*(1-H**2))/len(X)
        acc = beta*acc + (1-beta)*g**2
        V -= lr*g/(np.sqrt(acc)+1e-8)
    H = np.tanh(X @ V)
    U = rls(H)
    return V, U, np.asarray(costos)

def pca_svd(H, objetivo=0.99):
    mu = H.mean(axis=0)
    _, s, vt = np.linalg.svd(H-mu, full_matrices=False)
    if np.sum(s**2) == 0: raise ValueError("Sin variabilidad latente")
    acumulada = np.cumsum(s**2)/np.sum(s**2)
    k = min(len(s), int(np.searchsorted(acumulada, objetivo)+1))
    return mu, vt[:k].T, acumulada

def transformar(X, modelo):
    H = np.tanh(X @ modelo['V'])
    return (H-modelo['media_Xr']) @ modelo['Vpca']

def ajustar(X, J=20, objetivo=0.99, **kwargs):
    V,U,costos = entrenar_sls(X,J=J,**kwargs)
    mu,P,acum = pca_svd(np.tanh(X@V),objetivo)
    return dict(V=V,U=U,media_Xr=mu,Vpca=P), costos, acum


# ----------------------------------------------------------------------------
# Ejecución: python dimr.py  (SLS + PCA-SVD ajustados SOLO con Train)
# ----------------------------------------------------------------------------
J_SLS = 20
OBJETIVO_VARIANZA = 0.99


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from config import GRAFICOS, SALIDA, cargar_mde

    X, Y, metadata, F, tr, te = cargar_mde()
    modelo_dim, costos_dim, acum = ajustar(X[tr], J=J_SLS, objetivo=OBJETIVO_VARIANZA)
    Rtrain = transformar(X[tr], modelo_dim)
    np.savez(SALIDA / 'pesos.npz', **modelo_dim)
    pd.DataFrame({'epoca': np.arange(1, len(costos_dim) + 1), 'costo': costos_dim}) \
        .to_csv(SALIDA / 'costos_sls.csv', index=False)
    print('Dimensiones Train:', X[tr].shape, '→', (int(tr.sum()), J_SLS), '→', Rtrain.shape)

    plt.figure(); plt.plot(costos_dim)
    plt.xlabel('Época'); plt.ylabel('Costo regularizado (adimensional)'); plt.title('Convergencia SLS — Train')
    plt.grid(alpha=0.3); plt.savefig(GRAFICOS / 'convergencia_sls.png', dpi=150, bbox_inches='tight'); plt.close()

    plt.figure(); plt.plot(np.arange(1, len(acum) + 1), acum, marker='o')
    plt.axhline(OBJETIVO_VARIANZA, linestyle='--')
    plt.xlabel('Número de componentes'); plt.ylabel('Varianza explicada acumulada')
    plt.title('PCA-SVD sobre la representación SLS — Train'); plt.grid(alpha=0.3)
    plt.savefig(GRAFICOS / 'varianza_acumulada_pca.png', dpi=150, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
