"""
Configuración compartida del proyecto Auricle (reposo vs Valsalva).

TODOS los scripts viven en la misma carpeta que los registros WFDB (.hea/.dat)
y activity_labels.csv; los resultados se guardan en subcarpetas de esa carpeta.

Flujo:  eda.py -> mdef.py -> exploracion_mde.py -> dimr.py -> train.py -> test.py
        -> experimento_alt.py   (run_all.py ejecuta todo en orden)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# --- Rutas -------------------------------------------------------------------
RUTA = Path(__file__).resolve().parent / "data"            # carpeta con datos y scripts
EDA_DIR = "resultados_eda_HC2_HC14_sin_HC5"       # nombre usado por mdef.py
EDA = RUTA / EDA_DIR
SALIDA = RUTA / "experimento_integrado"
MDE = SALIDA / "mde"
GRAFICOS = SALIDA / "graficos"
for _d in (EDA, SALIDA, MDE, GRAFICOS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Experimento -------------------------------------------------------------
PARTICIPANTES = [f"HC_{i}_1" for i in range(2, 15) if i != 5]   # 12 sujetos
FS = 1000                      # Hz
T_VENTANA = 10                 # s
W = FS * T_VENTANA             # 10.000 muestras por ventana
SENALES = ["ECG", "PPG_Ear"]
CANAL_ACTIVIDAD = "Activity_Annotations"
CLASES = {0: "rest", 2: "valsalva"}              # código original -> nombre
LIMITES = {"ECG": (-5.0, 4.99985), "PPG_Ear": (-10.0, 9.99969)}  # candidatos empíricos
TOL = 1e-4
SEED = 42


def cargar_mde():
    """Lee las salidas de mdef.py y devuelve (X, Y, metadata, F, tr, te).

    X : MDE estandarizada con Train (n x 100)      <- dfeatures.csv
    Y : etiquetas one-hot (n x 2)                  <- dclass.csv
    F : MDE original sin estandarizar (DataFrame)  <- features_originales.csv
    tr, te : máscaras booleanas de Train / Test (partición por sujeto)
    """
    for nombre in ("dfeatures.csv", "dclass.csv", "metadata.csv", "features_originales.csv"):
        if not (MDE / nombre).exists():
            raise FileNotFoundError(f"Falta {MDE / nombre}. Ejecuta antes mdef.py")
    X = pd.read_csv(MDE / "dfeatures.csv", header=None, float_precision="round_trip").to_numpy(float)
    Y = pd.read_csv(MDE / "dclass.csv", header=None).to_numpy(int)
    metadata = pd.read_csv(MDE / "metadata.csv")
    F = pd.read_csv(MDE / "features_originales.csv", float_precision="round_trip")
    tr = metadata.Split.eq("train").to_numpy()
    te = ~tr
    assert len(X) == len(Y) == len(metadata) == len(F)
    assert not set(metadata.loc[tr, "Participante"]) & set(metadata.loc[te, "Participante"])
    return X, Y, metadata, F, tr, te
