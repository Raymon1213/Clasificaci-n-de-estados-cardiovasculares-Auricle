import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, accuracy_score
from train import predecir

def evaluar(X,Y,modelo):
    P,pred=predecir(X,modelo); y=Y.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=[0,1])
    reporte=pd.DataFrame(classification_report(y,pred,labels=[0,1],target_names=['rest','valsalva'],output_dict=True,zero_division=0)).T
    resumen=dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),f1_macro=f1_score(y,pred,average='macro',zero_division=0),f1_valsalva=f1_score(y,pred,zero_division=0))
    return P,pred,cm,reporte,resumen


# ----------------------------------------------------------------------------
# Ejecución: python test.py  (evaluación única sobre los 3 sujetos de Test)
# ----------------------------------------------------------------------------
def guardar_reportes_y_graficos(y_real, y_pred, probabilidades, cm, nombres_clases, base_dir):
    """Exporta F-scores, matriz de confusión, métricas por clase y predicciones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_real, y_pred, average=None, labels=range(len(nombres_clases)), zero_division=0)
    df_fscores = pd.DataFrame({'Clase': nombres_clases, 'F-score': fscore})
    df_fscores.loc[len(df_fscores)] = ['Promedio macro', np.mean(fscore)]
    df_fscores.to_csv(base_dir / 'fscores.csv', index=False)
    pd.DataFrame(cm).to_csv(base_dir / 'cmatriz.csv', index=False, header=False)
    pd.DataFrame(np.column_stack([precision, recall, fscore]),
                 columns=['Precision', 'Recall', 'F-score'], index=nombres_clases) \
        .to_csv(base_dir / 'metricas.csv')
    df_pred = pd.DataFrame({'Clase_real': y_real, 'Clase_predicha': y_pred})
    for i, nombre in enumerate(nombres_clases):
        df_pred[f'P_{nombre}'] = probabilidades[:, i]
    df_pred.to_csv(base_dir / 'predicciones.csv', index=False)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de confusión — Test'); plt.colorbar()
    pos = np.arange(len(nombres_clases))
    plt.xticks(pos, nombres_clases, rotation=45); plt.yticks(pos, nombres_clases)
    for i in range(len(nombres_clases)):
        for j in range(len(nombres_clases)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Clase predicha'); plt.ylabel('Clase real'); plt.tight_layout()
    plt.savefig(base_dir / 'matriz.png', dpi=300, bbox_inches='tight'); plt.close()
    print(df_fscores.round(4).to_string(index=False))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                                 balanced_accuracy_score, classification_report, f1_score, recall_score)
    from sklearn.metrics import confusion_matrix as cm_fn
    import dimr
    from config import GRAFICOS, SALIDA, cargar_mde

    X, Y, metadata, F, tr, te = cargar_mde()
    modelo_dim = dict(np.load(SALIDA / 'pesos.npz'))
    modelo_soft = dict(np.load(SALIDA / 'pesoftmax.npz'))

    # Mismos parámetros de Train aplicados a Test (sin reajustar nada)
    Rtest = dimr.transformar(X[te], modelo_dim)
    P, pred, cm, reporte, metricas = evaluar(Rtest, Y[te], modelo_soft)
    print(reporte.to_string()); print(metricas)

    # Referencia trivial: pesos nulos => predice siempre la misma clase (reposo)
    _, _, _, _, baseline = evaluar(Rtest, Y[te], dict(
        W=np.zeros_like(modelo_soft['W']), media=modelo_soft['media'],
        desviacion=modelo_soft['desviacion']))
    print('Baseline siempre reposo:', baseline)

    ConfusionMatrixDisplay(cm, display_labels=['Rest', 'Valsalva']).plot()
    plt.title('Test — sujetos no vistos')
    plt.savefig(GRAFICOS / 'matriz_confusion_test.png', dpi=150, bbox_inches='tight'); plt.close()

    resultados = metadata.loc[te].copy()
    resultados['Prediccion'] = pred
    resultados['P_valsalva'] = P[:, 1]
    resultados.to_csv(SALIDA / 'predicciones_test.csv', index=False)
    reporte.to_csv(SALIDA / 'reporte_test.csv')
    for sujeto, g in resultados.groupby('Participante'):
        print(sujeto, 'matriz:', cm_fn(g.Etiqueta, g.Prediccion, labels=[0, 1]).tolist())
    pd.DataFrame([metricas, baseline], index=['Softmax', 'Siempre reposo']) \
        .to_csv(SALIDA / 'metricas_test.csv')

    # Métricas y gráficos adicionales del modelo original
    y_real = Y[te].argmax(axis=1)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Rest', 'Valsalva']).plot(ax=ax[0], colorbar=False)
    ax[0].set_title('Matriz de confusión - Test original')
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=['Rest', 'Valsalva']) \
        .plot(ax=ax[1], colorbar=False, values_format='.2f')
    ax[1].set_title('Recall por clase')
    RocCurveDisplay.from_predictions(y_real, P[:, 1], ax=ax[2]); ax[2].set_title('Curva ROC')
    PrecisionRecallDisplay.from_predictions(y_real, P[:, 1], ax=ax[3])
    ax[3].axhline(y_real.mean(), ls='--', c='gray'); ax[3].set_title('Precisión-Recall')
    plt.tight_layout(); plt.savefig(GRAFICOS / 'metricas_test_original.png', dpi=150, bbox_inches='tight'); plt.close()

    print('Balanced accuracy original:', round(balanced_accuracy_score(y_real, pred), 3))
    print('F1 macro original:', round(f1_score(y_real, pred, average='macro'), 3))
    print('Recall de Valsalva original:', round(recall_score(y_real, pred, pos_label=1), 3))
    print(classification_report(y_real, pred, target_names=['Rest', 'Valsalva'], zero_division=0))

    filas = []
    for sujeto, g in resultados.groupby('Participante'):
        filas.append({'Participante': sujeto, 'N': len(g),
                      'Balanced_accuracy': balanced_accuracy_score(g.Etiqueta, g.Prediccion),
                      'F1_Valsalva': f1_score(g.Etiqueta, g.Prediccion, pos_label=1, zero_division=0)})
    tabla = pd.DataFrame(filas)
    print(tabla.round(3).to_string(index=False))
    tabla.to_csv(SALIDA / 'metricas_por_sujeto_test_adicionales.csv', index=False)

    guardar_reportes_y_graficos(y_real, pred, P, cm, ['rest', 'valsalva'], SALIDA)
    print('Resultados guardados en:', SALIDA)


if __name__ == '__main__':
    main()
