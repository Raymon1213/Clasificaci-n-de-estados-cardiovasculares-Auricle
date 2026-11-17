#!/usr/bin/env python3
"""
Experimento adicional (secciones finales del Colab): MDE + 16 características básicas
+ normalización robusta por sujeto, comparado con el modelo original.

- La selección se respalda con LOSO dentro de los 9 sujetos de Train.
- Test se usa UNA vez, después de fijar el método; no se eligen parámetros con Test.
- La normalización robusta usa mediana/IQR de todas las ventanas (sin etiquetas) del
  sujeto evaluado: escenario transductivo, no válido para una ventana aislada.

Requiere haber ejecutado antes: mdef.py y test.py (usa predicciones_test.csv).
Opción --recalcular: vuelve a calcular las características básicas (si no, usa la caché).
"""
import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy import signal, stats

import auricle_cv_alt as cv_alt
import auricle_test_alt as ev_alt
from config import FS, GRAFICOS, RUTA, SALIDA, SENALES, cargar_mde

BANDAS_HZ = [(0.5, 4.0), (4.0, 15.0), (15.0, 40.0)]


def nombres_basicas(senales):
    base = ["std", "ptp", "mad_diff", "asimetria", "curtosis"] + \
           [f"logP_{a:g}_{b:g}Hz" for a, b in BANDAS_HZ]
    return [f"{s}_{n}" for s in senales for n in base]


def features_basicas(ventana, fs=1000):
    """Amplitud, forma y potencia relativa por banda (Welch). La MDE es invariante a la
    amplitud (z-score por ventana); estas variables recuperan esa información."""
    ventana = np.asarray(ventana, dtype=float)
    out = []
    for x in ventana.T:
        f, p = signal.welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4096))
        total = p[(f >= 0.5) & (f <= 40)].sum() + 1e-20
        bandas = [np.log10(p[(f >= a) & (f < b)].sum() / total + 1e-12) for a, b in BANDAS_HZ]
        out += [x.std(), np.ptp(x), np.mean(np.abs(np.diff(x))),
                stats.skew(x), stats.kurtosis(x), *bandas]
    return np.asarray(out)


def calcular_basicas(metadata):
    basicas = np.empty((len(metadata), len(nombres_basicas(SENALES))), dtype=float)
    for sujeto, idx in metadata.groupby("Participante", sort=False).groups.items():
        registro = wfdb.rdrecord(os.path.join(RUTA, sujeto), channel_names=SENALES, physical=True)
        datos = np.asarray(registro.p_signal, dtype=float)
        for pos in idx:
            fila = metadata.loc[pos]
            a, b = int(fila.Inicio_muestra), int(fila.Fin_muestra)
            basicas[pos] = features_basicas(datos[a:b], fs=FS)
        print("Características básicas:", sujeto)
        del registro, datos
        gc.collect()
    assert np.isfinite(basicas).all()
    return basicas


def main(recalcular=False):
    X, Y, metadata, F, tr, te = cargar_mde()

    cache = SALIDA / "features_basicas_alternativa.csv"
    if cache.exists() and not recalcular:
        basicas = pd.read_csv(cache, float_precision="round_trip").to_numpy(float)
        assert basicas.shape == (len(metadata), len(nombres_basicas(SENALES)))
        print("Características básicas cargadas de caché:", cache.name)
    else:
        basicas = calcular_basicas(metadata)
        pd.DataFrame(basicas, columns=nombres_basicas(SENALES)).to_csv(cache, index=False)

    assert len(F) == len(metadata) == len(basicas)
    M_ALTERNATIVA = np.hstack([F.to_numpy(dtype=float), basicas])
    print("MDE:", F.shape, "| básicas:", basicas.shape, "| combinado:", M_ALTERNATIVA.shape)

    y_alt = metadata.Etiqueta.to_numpy(dtype=int)
    g_alt = metadata.Participante.to_numpy()
    train_sujetos = sorted(metadata.loc[tr, "Participante"].unique())

    # --- Comparación SOLO con los 9 sujetos de Train (LOSO): no se toca Test ---
    base_alt = cv_alt.loso(F.to_numpy(float), y_alt, g_alt, cv_alt.CFG_BASE, sujetos=train_sujetos)
    config_robusta = {**cv_alt.CFG_BASE, "norm_sujeto": "robusta"}
    robusta_alt = cv_alt.loso(M_ALTERNATIVA, y_alt, g_alt, config_robusta, sujetos=train_sujetos)
    tabla = pd.DataFrame({
        "Original MDE (LOSO-Train)": cv_alt.resumir(base_alt),
        "MDE + básicas + robusta (LOSO-Train)": cv_alt.resumir(robusta_alt),
    }).T
    columnas = ["balanced_accuracy", "f1_macro", "f1_valsalva", "recall_valsalva", "roc_auc", "pr_auc"]
    print(tabla[columnas].round(3).to_string())
    tabla.to_csv(SALIDA / "comparacion_loso_train_selectiva.csv")
    tabla[columnas[:4]].T.plot.bar(figsize=(11, 4), rot=25)
    plt.ylim(0, 1); plt.ylabel("Métrica (adimensional)")
    plt.title("LOSO-Train: comparación de metodología"); plt.tight_layout()
    plt.savefig(GRAFICOS / "comparacion_loso_train.png", dpi=150, bbox_inches="tight"); plt.close()

    # --- Evaluación final en los 3 sujetos de Test (una sola vez) ---
    archivo = SALIDA / "predicciones_test.csv"
    if not archivo.exists():
        raise FileNotFoundError("Falta predicciones_test.csv: ejecuta antes test.py")
    orig = pd.read_csv(archivo, float_precision="round_trip")
    assert (orig.Participante.to_numpy() == g_alt[te]).all(), "Orden de ventanas inconsistente"

    modelo_alt = cv_alt.ajustar_pipeline(M_ALTERNATIVA[tr], y_alt[tr], config_robusta, g_alt[tr])
    proba_alt = cv_alt.predecir_pipeline(M_ALTERNATIVA[te], modelo_alt, g_alt[te])
    y_test = y_alt[te]
    pred_alt = (proba_alt[:, 1] > 0.5).astype(int)
    met_alt = ev_alt.metricas(y_test, proba_alt[:, 1], pred_alt)
    met_orig = ev_alt.metricas(y_test, orig.P_valsalva.to_numpy(), orig.Prediccion.to_numpy())
    comp = pd.DataFrame({"Original": met_orig, "Alternativa robusta": met_alt}).T
    print(comp[["accuracy", "balanced_accuracy", "f1_macro", "f1_valsalva",
                "recall_valsalva", "roc_auc", "pr_auc"]].round(3).to_string())
    comp.to_csv(SALIDA / "comparacion_test_original_vs_robusta.csv")
    print("Diferencia balanced accuracy alternativa - original:",
          round(met_alt["balanced_accuracy"] - met_orig["balanced_accuracy"], 4))
    print("Una mejora en LOSO-Train por sí sola no demuestra mejora en Test.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recalcular", action="store_true")
    main(**vars(ap.parse_args()))
