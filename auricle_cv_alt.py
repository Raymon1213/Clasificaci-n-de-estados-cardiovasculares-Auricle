"""Validación cruzada dejando un sujeto fuera (LOSO) para toda la cadena
(estandarización → SLS → PCA → Softmax). TODO lo que se ajusta (escaladores, SLS, PCA, Softmax)
se reajusta en cada pliegue solo con los sujetos de entrenamiento; la MDE no se reajusta porque
se calcula ventana a ventana sin usar etiquetas ni otros sujetos."""
import time

import numpy as np
import pandas as pd

import auricle_dimr_alt as dimr
import auricle_train_alt as train
import auricle_test_alt as ev

ARQUITECTURAS = {                       # (usa_sls, usa_pca)
    'mde_softmax': (False, False),
    'mde_pca_softmax': (False, True),
    'mde_sls_softmax': (True, False),
    'mde_sls_pca_softmax': (True, True),
}
CFG_BASE = dict(arquitectura='mde_sls_pca_softmax', J=20, objetivo=0.99, k_pca=None,
                alpha=1.0, l2=1e-3, epochs_sls=500, lr_sls=1e-3, epochs_soft=1000, lr_soft=1e-2,
                norm_sujeto=None, seed=42)


def _norm_sujeto(M, grupos, modo):
    """Normalización por sujeto SIN etiquetas (reduce el desplazamiento entre personas).
    'centrar' resta la media; 'z' media/desv.; 'robusta' mediana/IQR (insensible a la maniobra)."""
    if modo is None:
        return np.asarray(M, float)
    if grupos is None:
        raise ValueError('norm_sujeto requiere el vector de grupos')
    M = np.array(M, float)
    g = np.asarray(grupos)
    for s in np.unique(g):
        i = g == s
        if modo == 'centrar':
            c, e = M[i].mean(0), 1.0
        elif modo == 'z':
            c, e = M[i].mean(0), M[i].std(0)
        elif modo == 'robusta':
            q1, med, q3 = np.percentile(M[i], [25, 50, 75], axis=0)
            c, e = med, (q3 - q1) / 1.349
        else:
            raise ValueError('norm_sujeto desconocido')
        e = np.where(np.asarray(e) < 1e-12, 1.0, e)
        M[i] = (M[i] - c) / e
    return M


def ajustar_pipeline(M, y, cfg=None, grupos=None):
    cfg = {**CFG_BASE, **(cfg or {})}
    usa_sls, usa_pca = ARQUITECTURAS[cfg['arquitectura']]
    M = _norm_sujeto(M, grupos, cfg['norm_sujeto'])
    mu, sd = M.mean(0), M.std(0)
    sd = np.where(sd < 1e-12, 1, sd)
    Z = (M - mu) / sd
    dim, _, acum = dimr.ajustar(Z, J=min(cfg['J'], Z.shape[1] - 1), objetivo=cfg['objetivo'],
                                sls=usa_sls, pca=usa_pca, k=cfg['k_pca'],
                                epochs=cfg['epochs_sls'], lr=cfg['lr_sls'], seed=cfg['seed'])
    R = dimr.transformar(Z, dim)
    Y = np.eye(2)[np.asarray(y, int)]
    soft, costos = train.entrenar(R, Y, epochs=cfg['epochs_soft'], lr=cfg['lr_soft'],
                                  l2=cfg['l2'], alpha=cfg['alpha'])
    return dict(cfg=cfg, mu=mu, sd=sd, dim=dim, soft=soft, costos=costos, acum=acum)


def predecir_pipeline(M, modelo, grupos=None):
    M = _norm_sujeto(M, grupos, modelo['cfg']['norm_sujeto'])
    R = dimr.transformar((M - modelo['mu']) / modelo['sd'], modelo['dim'])
    P, _ = train.predecir(R, modelo['soft'])
    return P


def loso(M, y, grupos, cfg=None, sujetos=None, permutar=False, semilla_perm=0):
    """Cada sujeto de `sujetos` se deja fuera una vez; se entrena con los demás de ese conjunto.
    permutar=True baraja las etiquetas dentro de cada sujeto de entrenamiento (control de azar)."""
    M, y, g = np.asarray(M, float), np.asarray(y, int), np.asarray(grupos)
    pool = list(sujetos) if sujetos is not None else list(pd.unique(g))
    rng = np.random.default_rng(semilla_perm)
    salida = []
    for s in pool:
        te = g == s
        tr = np.isin(g, pool) & ~te
        ytr = y[tr].copy()
        if permutar:
            gtr = g[tr]
            for t in np.unique(gtr):
                j = np.flatnonzero(gtr == t)
                ytr[j] = rng.permutation(ytr[j])
        mod = ajustar_pipeline(M[tr], ytr, cfg, g[tr])
        P = predecir_pipeline(M[te], mod, g[te])
        salida.append(pd.DataFrame(dict(Participante=s, idx=np.flatnonzero(te),
                                        Etiqueta=y[te], P_valsalva=P[:, 1])))
    return pd.concat(salida, ignore_index=True)


def resumir(df_pred, umbral=0.5):
    """Métricas agrupadas (todas las ventanas OOF) + promedio por sujeto."""
    pred = (df_pred.P_valsalva.to_numpy() > umbral).astype(int)
    r = ev.metricas(df_pred.Etiqueta.to_numpy(), df_pred.P_valsalva.to_numpy(), pred)
    d = df_pred.assign(Prediccion=pred)
    ps = ev.por_sujeto(d)
    r['bal_acc_media_sujetos'] = ps.balanced_accuracy.mean()
    r['bal_acc_std_sujetos'] = ps.balanced_accuracy.std()
    r['auc_media_sujetos'] = ps.roc_auc.mean()
    return r


def _cm_por_sujeto(df_pred, umbral):
    pred = (df_pred.P_valsalva.to_numpy() > umbral).astype(int)
    d = df_pred.assign(Prediccion=pred)
    ps = ev.por_sujeto(d)
    return ps[['TN', 'FP', 'FN', 'TP']].to_numpy(float)


def ic_bootstrap(df_pred, umbral=0.5, n=2000, seed=0):
    """Intervalo bootstrap remuestreando SUJETOS (no ventanas: las ventanas de un sujeto
    no son independientes). Con pocos sujetos el intervalo será ancho: es lo honesto."""
    cm = _cm_por_sujeto(df_pred, umbral)
    rng = np.random.default_rng(seed)
    bal, f1 = [], []
    for _ in range(n):
        tn, fp, fn, tp = cm[rng.integers(0, len(cm), len(cm))].sum(0)
        rec_v = tp / (tp + fn) if tp + fn else np.nan
        rec_r = tn / (tn + fp) if tn + fp else np.nan
        bal.append(np.nanmean([rec_v, rec_r]))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
    q = lambda v: np.nanpercentile(v, [2.5, 50, 97.5])
    return dict(balanced_accuracy=q(bal), f1_valsalva=q(f1))


def buscar_umbral(df_pred, criterio='balanced_accuracy', grilla=None):
    """Umbral que maximiza el criterio sobre predicciones fuera de muestra (OOF).
    Si se evalúa sobre los mismos datos con que se eligió, el resultado es optimista:
    para una estimación honesta usar loso_anidado."""
    grilla = np.linspace(0.05, 0.95, 91) if grilla is None else grilla
    y, p = df_pred.Etiqueta.to_numpy(), df_pred.P_valsalva.to_numpy()
    mejor, u_mejor = -np.inf, 0.5
    for u in grilla:
        pred = (p > u).astype(int)
        tp, fn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 1))
        tn, fp = np.sum((pred == 0) & (y == 0)), np.sum((pred == 1) & (y == 0))
        if criterio == 'balanced_accuracy':
            v = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        else:
            v = 2 * tp / max(2 * tp + fp + fn, 1)
        if v > mejor:
            mejor, u_mejor = v, float(u)
    return u_mejor, mejor


def tabla_experimentos(experimentos, y, grupos, sujetos=None, umbral=0.5, verbose=True):
    """experimentos: {nombre: (matriz_de_variables, cfg)} → tabla ordenada por balanced accuracy."""
    filas, preds = {}, {}
    for nombre, (M, cfg) in experimentos.items():
        t0 = time.time()
        p = loso(M, y, grupos, cfg, sujetos)
        preds[nombre] = p
        filas[nombre] = resumir(p, umbral)
        if verbose:
            print(f'{nombre:<40s} bal.acc={filas[nombre]["balanced_accuracy"]:.3f}  ({time.time()-t0:.0f} s)', flush=True)
    tabla = pd.DataFrame(filas).T.sort_values('balanced_accuracy', ascending=False)
    return tabla, preds


def loso_anidado(M, y, grupos, candidatas, sujetos=None, criterio='balanced_accuracy', ajustar_umbral=True):
    """Selección de hiperparámetros (y umbral) DENTRO de cada pliegue externo.
    candidatas: {nombre: cfg}. Costo ≈ n_sujetos × (n_sujetos-1) × n_candidatas ajustes."""
    M, y, g = np.asarray(M, float), np.asarray(y, int), np.asarray(grupos)
    pool = list(sujetos) if sujetos is not None else list(pd.unique(g))
    salida = []
    for s in pool:
        interno = [t for t in pool if t != s]
        puntajes, umbrales = {}, {}
        for nombre, cfg in candidatas.items():
            p = loso(M, y, g, cfg, interno)
            u, v = buscar_umbral(p, criterio) if ajustar_umbral else (0.5, resumir(p)[criterio])
            puntajes[nombre], umbrales[nombre] = v, u
        mejor = max(puntajes, key=puntajes.get)
        tr, te = np.isin(g, interno), g == s
        mod = ajustar_pipeline(M[tr], y[tr], candidatas[mejor], g[tr])
        P = predecir_pipeline(M[te], mod, g[te])
        salida.append(pd.DataFrame(dict(Participante=s, idx=np.flatnonzero(te), Etiqueta=y[te],
                                        P_valsalva=P[:, 1], Elegida=mejor, Umbral=umbrales[mejor])))
        print(f'{s}: elegida={mejor} umbral={umbrales[mejor]:.2f}', flush=True)
    d = pd.concat(salida, ignore_index=True)
    d['Prediccion'] = (d.P_valsalva > d.Umbral).astype(int)
    return d
