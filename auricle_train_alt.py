"""Regresión Softmax con entropía cruzada ponderada + L2, optimizada con RMSprop."""
import numpy as np
from scipy.special import logsumexp


def softmax(Z):
    return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


def entrenar(X, Y, epochs=1000, lr=0.01, beta=0.9, l2=0.001, ponderar=True,
             alpha=1.0, eps=1e-8, tol=1e-7, paciencia=50):
    """alpha: intensidad del peso de clase  w_k = (N / (K N_k))^alpha
              (0 = sin ponderar, 1 = balanceo completo, como en el Experimento 1).
    tol: parada temprana si el costo cambia < tol durante `paciencia` épocas (None = desactivada)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1, sd)
    A = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    counts = Y.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Falta una clase en Train")
    pesos = (len(Y) / (2 * counts)) ** alpha if ponderar else np.ones(2)
    w = Y @ pesos
    W = np.zeros((A.shape[1], 2))
    acc = np.zeros_like(W)
    costos = []
    for ep in range(epochs):
        Z = A @ W
        logp = Z - logsumexp(Z, axis=1, keepdims=True)
        P = np.exp(logp)
        costos.append(-np.sum(w[:, None] * Y * logp) / w.sum() + l2 * np.sum(W[:-1] ** 2) / 2)
        if tol is not None and ep > paciencia and abs(costos[-paciencia - 1] - costos[-1]) < tol:
            break
        g = A.T @ (w[:, None] * (P - Y)) / w.sum()
        g[:-1] += l2 * W[:-1]                      # el sesgo no se penaliza
        acc = beta * acc + (1 - beta) * g ** 2
        W -= lr * g / (np.sqrt(acc) + eps)
    return dict(W=W, media=mu, desviacion=sd, pesos_clase=pesos, epocas=len(costos)), np.asarray(costos)


def predecir(X, modelo, umbral=None):
    """umbral=None → argmax (equivale a P(valsalva)>0.5); si se da, pred = P(valsalva) > umbral."""
    A = np.column_stack([(X - modelo['media']) / modelo['desviacion'], np.ones(len(X))])
    P = softmax(A @ modelo['W'])
    pred = P.argmax(axis=1) if umbral is None else (P[:, 1] > umbral).astype(int)
    return P, pred
