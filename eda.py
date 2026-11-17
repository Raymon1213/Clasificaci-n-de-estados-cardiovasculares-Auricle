#!/usr/bin/env python3
"""
EDA + segmentación (secciones 1 y 2 del Colab).

Entradas : registros WFDB HC_*_1.hea/.dat y activity_labels.csv (misma carpeta).
Salidas  : resultados_eda_HC2_HC14_sin_HC5/ventanas_finales.csv   (lo usa mdef.py)
           experimento_integrado/detalle_ventanas.csv, resumen_eda.csv
           experimento_integrado/graficos/*.png
"""
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from config import (CANAL_ACTIVIDAD, CLASES, EDA, FS, GRAFICOS, LIMITES,
                    PARTICIPANTES, RUTA, SALIDA, SENALES, TOL, W)


# ----------------------------------------------------------------------------
# Lectura y decodificación
# ----------------------------------------------------------------------------
def cargar_labels():
    df = pd.read_csv(RUTA / "activity_labels.csv")
    return df, dict(zip(df["activity_number"].astype(int), df["activity_name"].str.strip()))


def cargar_participante(participante, labels_map):
    """Devuelve (X[N,2] = ECG, PPG_Ear; activity[N] enteros; fs) tras validar el registro.

    Las anotaciones se decodifican desde valores FÍSICOS y se aceptan solo si son
    enteros (atol 1e-5) y pertenecen a activity_labels.csv; si no, se aborta."""
    base = os.path.join(RUTA, participante)
    header = wfdb.rdheader(base)
    canales = SENALES + [CANAL_ACTIVIDAD]

    if header.fs != FS:
        raise ValueError(f"{participante}: frecuencia inesperada {header.fs} Hz")
    for canal in canales:
        if canal not in header.sig_name:
            raise ValueError(f"{participante}: falta el canal {canal}")

    record = wfdb.rdrecord(base, channel_names=canales, physical=True)
    X = record.p_signal[:, :2]
    activity_raw = record.p_signal[:, 2]

    if not np.all(np.isfinite(activity_raw)):
        raise ValueError(f"{participante}: anotaciones no finitas")
    activity = np.rint(activity_raw).astype(int)
    if not np.allclose(activity_raw, activity, atol=1e-5, rtol=0):
        raise ValueError(f"{participante}: las anotaciones no corresponden a códigos enteros")

    desconocidos = set(np.unique(activity)) - set(labels_map.keys())
    if desconocidos:
        raise ValueError(f"{participante}: códigos desconocidos {desconocidos}")
    if len(X) != header.sig_len:
        raise ValueError(f"{participante}: longitud inconsistente")
    return X, activity, header.fs


def obtener_periodos(activity, labels_map):
    """Períodos continuos de una misma actividad (inicio, fin, duración, ventanas de 10 s)."""
    activity = np.asarray(activity)
    columnas = ["Codigo", "Actividad", "Inicio_muestra", "Fin_muestra", "Muestras",
                "Inicio_s", "Fin_s", "Duracion_s", "Ventanas_10s"]
    if len(activity) == 0:
        return pd.DataFrame(columns=columnas)
    cambios = np.where(activity[1:] != activity[:-1])[0] + 1
    inicios = np.r_[0, cambios]
    finales = np.r_[cambios, len(activity)]
    periodos = []
    for inicio, fin in zip(inicios, finales):
        codigo = int(activity[inicio])
        muestras = fin - inicio
        periodos.append({
            "Codigo": codigo, "Actividad": labels_map[codigo],
            "Inicio_muestra": inicio, "Fin_muestra": fin, "Muestras": muestras,
            "Inicio_s": inicio / FS, "Fin_s": fin / FS, "Duracion_s": muestras / FS,
            "Ventanas_10s": muestras // W,
        })
    return pd.DataFrame(periodos)


# ----------------------------------------------------------------------------
# Calidad
# ----------------------------------------------------------------------------
def mayor_tramo_constante(x):
    """Longitud (muestras) del tramo más largo de valor constante."""
    x = np.asarray(x)
    if len(x) == 0:
        return 0
    cambios = np.where(x[1:] != x[:-1])[0] + 1
    limites = np.r_[0, cambios, len(x)]
    return int(np.diff(limites).max())


def mascara_limites(x, nombre):
    inferior, superior = LIMITES[nombre]
    return (np.isclose(x, inferior, atol=TOL, rtol=0) |
            np.isclose(x, superior, atol=TOL, rtol=0))


def mayor_episodio(mascara):
    bordes = np.flatnonzero(np.diff(np.r_[False, mascara, False].astype(int)))
    return int(np.max(bordes[1::2] - bordes[::2])) if len(bordes) else 0


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def guardar(nombre):
    plt.savefig(GRAFICOS / nombre, dpi=150, bbox_inches="tight")
    plt.close()


def graficar_senales(X, activity, participante, codigo, labels_map, unidades, duracion=10):
    periodos = obtener_periodos(activity, labels_map)
    sel = periodos[(periodos["Codigo"] == codigo) & (periodos["Duracion_s"] >= duracion)]
    if sel.empty:
        print("No hay un período suficientemente largo.")
        return
    inicio = int(sel.iloc[0]["Inicio_muestra"])
    segmento = X[inicio:inicio + int(duracion * FS)]
    tiempo = np.arange(len(segmento)) / FS
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for i, nombre in enumerate(SENALES):
        axes[i].plot(tiempo, segmento[:, i], linewidth=0.8)
        axes[i].set_title(nombre)
        axes[i].set_ylabel(f"Amplitud ({unidades[nombre]})")
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Tiempo (segundos)")
    fig.suptitle(f"{participante} - {labels_map[codigo]}")
    plt.tight_layout()
    guardar(f"senales_{participante}_{labels_map[codigo]}.png")


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    print("Ruta:", RUTA)
    print("Participantes:", PARTICIPANTES)
    print("Frecuencia:", FS, "Hz | Tamaño de ventana:", W, "muestras")

    # 1. EDA breve: población y duración --------------------------------------
    filas = []
    for p in PARTICIPANTES:
        h = wfdb.rdheader(os.path.join(RUTA, p))
        filas.append({
            "Participante": p, "Muestras": h.sig_len, "Frecuencia_Hz": h.fs,
            "Canales": h.n_sig, "Duracion_seg": h.sig_len / h.fs,
            "Duracion_min": h.sig_len / h.fs / 60,
            "ECG_disponible": "ECG" in h.sig_name,
            "PPG_Ear_disponible": "PPG_Ear" in h.sig_name,
            "Actividad_disponible": CANAL_ACTIVIDAD in h.sig_name,
        })
    df_registros = pd.DataFrame(filas)
    print(df_registros.round(3).to_string(index=False))
    print("\nResumen de duración (min):\n", df_registros["Duracion_min"].describe().round(3))
    print("\nTotal de muestras:", df_registros["Muestras"].sum())
    print("Duración total en horas:", df_registros["Duracion_seg"].sum() / 3600)

    assert len(df_registros) == 12
    assert (df_registros["Frecuencia_Hz"] == FS).all()
    assert df_registros[["ECG_disponible", "PPG_Ear_disponible", "Actividad_disponible"]].all().all()

    plt.figure(figsize=(13, 5))
    plt.bar(df_registros["Participante"], df_registros["Duracion_min"],
            color="steelblue", edgecolor="black")
    plt.axhline(df_registros["Duracion_min"].mean(), color="red", linestyle="--",
                label="Duración promedio")
    plt.title("Duración de los registros fisiológicos")
    plt.xlabel("Participante"); plt.ylabel("Duración (minutos)")
    plt.xticks(rotation=45); plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); guardar("duracion_registros.png")

    df_labels, labels_map = cargar_labels()
    print(df_labels.to_string(index=False))
    print("Diccionario de actividades:", labels_map)

    # Ejemplo de señales (primer sujeto): unidades tomadas del encabezado
    p0 = PARTICIPANTES[0]
    h0 = wfdb.rdheader(os.path.join(RUTA, p0))
    unidades = {s: h0.units[h0.sig_name.index(s)] for s in SENALES}
    X0, act0, _ = cargar_participante(p0, labels_map)
    for codigo in (0, 2):
        graficar_senales(X0, act0, p0, codigo, labels_map, unidades)
    del X0, act0

    # 2. Segmentación y calidad -------------------------------------------------
    resultados_ventanas = []
    for participante in PARTICIPANTES:
        print("Evaluando ventanas:", participante)
        X_i, activity_i, _ = cargar_participante(participante, labels_map)
        periodos_i = obtener_periodos(activity_i, labels_map)
        periodos_i = periodos_i[periodos_i["Codigo"].isin([0, 2])]

        for _, periodo in periodos_i.iterrows():
            inicio, fin = int(periodo["Inicio_muestra"]), int(periodo["Fin_muestra"])
            codigo = int(periodo["Codigo"])
            # Ventanas sin solapamiento DENTRO de cada período; se descarta el sobrante
            for inicio_w in range(inicio, fin - W + 1, W):
                fin_w = inicio_w + W
                ventana = X_i[inicio_w:fin_w]
                no_finitas = np.sum(~np.isfinite(ventana))
                std_ecg, std_ppg = np.std(ventana[:, 0]), np.std(ventana[:, 1])
                constante_ecg = mayor_tramo_constante(ventana[:, 0]) / FS
                constante_ppg = mayor_tramo_constante(ventana[:, 1]) / FS
                valida_basica = (no_finitas == 0 and std_ecg > 0 and std_ppg > 0)

                assert ventana.shape == (W, len(SENALES))
                assert np.all(activity_i[inicio_w:fin_w] == codigo)
                indicadores = {}
                for j, nombre in enumerate(SENALES):
                    mascara = mascara_limites(ventana[:, j], nombre)
                    indicadores[f"Saturacion_{nombre}_pct"] = 100 * mascara.mean()
                    indicadores[f"Mayor_episodio_limite_{nombre}_s"] = mayor_episodio(mascara) / FS

                resultados_ventanas.append({
                    **indicadores,
                    "Participante": participante, "Actividad": labels_map[codigo],
                    "Etiqueta": 0 if codigo == 0 else 1,
                    "Inicio_muestra": inicio_w, "Fin_muestra": fin_w,
                    "Inicio_s": inicio_w / FS, "Fin_s": fin_w / FS,
                    "No_finitas": no_finitas, "STD_ECG": std_ecg, "STD_PPG": std_ppg,
                    "Tramo_constante_ECG_s": constante_ecg,
                    "Tramo_constante_PPG_s": constante_ppg,
                    "Valida_basica": valida_basica,
                })
        del X_i, activity_i
        gc.collect()

    df_ventanas = pd.DataFrame(resultados_ventanas)
    print(df_ventanas.head(10).to_string())
    print("Total de ventanas:", len(df_ventanas))
    print("Ventanas válidas según criterio básico:", df_ventanas["Valida_basica"].sum())
    print("Ventanas con problemas básicos:", (~df_ventanas["Valida_basica"]).sum())

    # Exclusión conservadora: ventanas con muestras en los límites de saturación
    cols_sat = [f"Saturacion_{s}_pct" for s in SENALES]
    df_ventanas["Posible_saturacion"] = df_ventanas[cols_sat].gt(0).any(axis=1)
    df_final = df_ventanas.loc[df_ventanas.Valida_basica & ~df_ventanas.Posible_saturacion] \
        .copy().reset_index(drop=True)
    df_final.insert(0, "Ventana_ID", np.arange(len(df_final)))

    assert set(df_final.Participante) == set(PARTICIPANTES)
    assert (df_final.groupby("Participante").Etiqueta.nunique() == 2).all()
    assert not df_final.duplicated(["Participante", "Inicio_muestra"]).any()

    resumen = pd.crosstab(df_final.Participante, df_final.Actividad).reindex(PARTICIPANTES)
    print(resumen)
    print("Antes:", len(df_ventanas), "Después:", len(df_final),
          "Excluidas:", len(df_ventanas) - len(df_final))
    print(df_ventanas.loc[~df_ventanas.Valida_basica | df_ventanas.Posible_saturacion].to_string())

    resumen.plot.bar(figsize=(12, 4))
    plt.ylabel("Ventanas"); plt.title("Ventanas por participante y actividad")
    plt.tight_layout(); guardar("ventanas_por_participante.png")

    df_final.to_csv(EDA / "ventanas_finales.csv", index=False)
    df_ventanas.to_csv(SALIDA / "detalle_ventanas.csv", index=False)
    resumen.to_csv(SALIDA / "resumen_eda.csv")

    # Línea de tiempo de ventanas por clase
    fig, ax = plt.subplots(figsize=(13, 5))
    estilo = {0: ("tab:blue", CLASES[0]), 1: ("tab:orange", CLASES[2])}
    primero_exc = next((p for p in PARTICIPANTES
                        if (~df_ventanas[df_ventanas.Participante == p].Valida_basica |
                            df_ventanas[df_ventanas.Participante == p].Posible_saturacion).any()), None)
    for k, p in enumerate(PARTICIPANTES):
        d = df_ventanas[df_ventanas.Participante == p]
        for etq, (color, nombre) in estilo.items():
            s = d[d.Etiqueta == etq]
            ax.scatter(s.Inicio_s / 60, [k] * len(s), s=16, color=color,
                       label=nombre if k == 0 else None)
        exc = d[~d.Valida_basica | d.Posible_saturacion]
        ax.scatter(exc.Inicio_s / 60, [k] * len(exc), marker="x", color="red", s=45,
                   label="excluida" if p == primero_exc else None)
    ax.set_yticks(range(len(PARTICIPANTES))); ax.set_yticklabels(PARTICIPANTES)
    ax.set(xlabel="Tiempo de sesión (min)", title="Posición temporal de las ventanas por sujeto")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    plt.tight_layout(); guardar("linea_tiempo_ventanas.png")
    print("Listo. ventanas_finales.csv en:", EDA)


if __name__ == "__main__":
    main()
