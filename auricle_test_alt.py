"""Evaluación: métricas globales, por clase, por participante y baseline trivial."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
import auricle_train_alt as train


def metricas(y, p1, pred):
    """y, pred ∈ {0,1}; p1 = P(valsalva). Las métricas independientes del umbral (AUC) son
    las más estables cuando hay pocos sujetos y desbalance."""
    y, pred = np.asarray(y), np.asarray(pred)
    dos = len(np.unique(y)) == 2
    return dict(
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        f1_macro=f1_score(y, pred, average='macro', zero_division=0),
        f1_valsalva=f1_score(y, pred, zero_division=0),
        recall_valsalva=recall_score(y, pred, zero_division=0),
        precision_valsalva=precision_score(y, pred, zero_division=0),
        mcc=matthews_corrcoef(y, pred),
        kappa=cohen_kappa_score(y, pred),
        roc_auc=roc_auc_score(y, p1) if dos else np.nan,
        pr_auc=average_precision_score(y, p1) if dos else np.nan,
    )


def evaluar(X, Y, modelo, umbral=None):
    P, pred = train.predecir(X, modelo, umbral)
    y = Y.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    reporte = pd.DataFrame(classification_report(
        y, pred, labels=[0, 1], target_names=['rest', 'valsalva'],
        output_dict=True, zero_division=0)).T
    return P, pred, cm, reporte, metricas(y, P[:, 1], pred)


def baseline_siempre_reposo(y):
    y = np.asarray(y)
    return metricas(y, np.zeros(len(y)), np.zeros(len(y), dtype=int))


def por_sujeto(df, col_pred='Prediccion', col_p='P_valsalva'):
    """df con Participante, Etiqueta, Prediccion, P_valsalva → una fila por sujeto."""
    filas = []
    for s, g in df.groupby('Participante'):
        tn, fp, fn, tp = confusion_matrix(g.Etiqueta, g[col_pred], labels=[0, 1]).ravel()
        rr = tn / (tn + fp) if tn + fp else np.nan
        rv = tp / (tp + fn) if tp + fn else np.nan
        auc = roc_auc_score(g.Etiqueta, g[col_p]) if g.Etiqueta.nunique() == 2 else np.nan
        filas.append(dict(Participante=s, n_rest=tn + fp, n_valsalva=tp + fn, TN=tn, FP=fp, FN=fn, TP=tp,
                          accuracy=(tn + tp) / len(g), recall_rest=rr, recall_valsalva=rv,
                          balanced_accuracy=np.nanmean([rr, rv]), roc_auc=auc))
    return pd.DataFrame(filas).set_index('Participante')
