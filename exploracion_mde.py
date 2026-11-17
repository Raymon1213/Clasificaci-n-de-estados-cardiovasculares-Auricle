#!/usr/bin/env python3
"""
Exploración de la MDE (secciones 4 del Colab) — SOLO Train, no modifica el modelo.

Genera: curvas medias por clase, medias por sujeto y diferencias pareadas,
correlación entre variables, PCA 2D y η² (sujeto vs clase).
Salidas: experimento_integrado/graficos/*.png y tablas/eta2_pca_mde.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GRAFICOS, MDE, SALIDA, SENALES, cargar_mde


def guardar(nombre):
    plt.savefig(GRAFICOS / nombre, dpi=150, bbox_inches="tight")
    plt.close()


def eta2(v, grupos):
    """Fracción de la varianza de v explicada por los grupos (0-1)."""
    total = np.sum((v - v.mean()) ** 2)
    entre = sum(np.sum(grupos == g) * (v[grupos == g].mean() - v.mean()) ** 2
                for g in np.unique(grupos))
    return entre / total


def main():
    X, Y, metadata, F, tr, te = cargar_mde()
    tau_max = json.loads((MDE / "config.json").read_text(encoding="utf-8"))["tau_max"]
    esc = np.arange(1, tau_max + 1)
    met_tr = metadata.loc[tr]
    suj_train = sorted(met_tr.Participante.unique())

    # Curvas medias por clase (todas las ventanas de Train)
    for s in SENALES:
        cols = [f"{s}_MDE_{t:02d}" for t in esc]
        plt.figure(figsize=(8, 4))
        for a in ["rest", "valsalva"]:
            v = F.loc[tr & metadata.Actividad.eq(a), cols]
            plt.plot(esc, v.mean().to_numpy(), label=a)
        plt.xlabel("Escala temporal τ (muestras a 1 kHz)"); plt.ylabel("Entropía normalizada")
        plt.title(f"MDE media por clase — {s} (Train)"); plt.legend(); plt.grid()
        guardar(f"mde_curvas_clase_{s}.png")

    # Medias por sujeto y diferencias pareadas
    def curvas_por_sujeto(canal):
        columnas = [f"{canal}_MDE_{t:02d}" for t in esc]
        reposo, valsalva = [], []
        for sujeto, indices in met_tr.groupby("Participante").groups.items():
            filas = F.loc[indices]
            etiquetas = met_tr.loc[indices, "Etiqueta"]
            reposo.append(filas.loc[etiquetas.eq(0), columnas].mean().to_numpy())
            valsalva.append(filas.loc[etiquetas.eq(1), columnas].mean().to_numpy())
        return np.array(reposo), np.array(valsalva)

    fig, axes = plt.subplots(len(SENALES), 2, figsize=(13, 7))
    for k, canal in enumerate(SENALES):
        R, V = curvas_por_sujeto(canal)
        for nombre, datos in [("Reposo", R), ("Valsalva", V)]:
            media = datos.mean(axis=0)
            ee = datos.std(axis=0, ddof=1) / np.sqrt(len(datos))
            axes[k, 0].plot(esc, media, label=nombre)
            axes[k, 0].fill_between(esc, media - ee, media + ee, alpha=.2)
        axes[k, 0].set(title=f"{canal}: promedio entre sujetos ± EE",
                       xlabel="Escala τ", ylabel="MDE normalizada")
        axes[k, 0].legend(); axes[k, 0].grid(alpha=.3)
        D = V - R
        for fila in D:
            axes[k, 1].plot(esc, fila, c="gray", alpha=.35)
        axes[k, 1].plot(esc, D.mean(axis=0), c="black", lw=2, label="Promedio")
        axes[k, 1].axhline(0, color="red", ls="--")
        axes[k, 1].set(title=f"{canal}: diferencia Valsalva - Reposo por sujeto",
                       xlabel="Escala τ", ylabel="Δ MDE")
        axes[k, 1].grid(alpha=.3)
    plt.tight_layout(); guardar("mde_por_sujeto.png")

    # Redundancia y variación sujeto vs clase
    Ftr = F.loc[tr]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    im = axes[0].imshow(Ftr.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].axhline(tau_max - 0.5, color="k", linewidth=0.8)
    axes[0].axvline(tau_max - 0.5, color="k", linewidth=0.8)
    axes[0].set_title("Correlación MDE — Train (bloques: ECG | PPG_Ear)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    Zc = ((Ftr - Ftr.mean()) / Ftr.std()).to_numpy()
    U_, s_, _ = np.linalg.svd(Zc, full_matrices=False)
    PC = U_[:, :3] * s_[:3]
    suj = met_tr["Participante"].to_numpy()
    act = met_tr["Etiqueta"].to_numpy()
    colores = plt.cm.tab10(np.arange(len(suj_train)))
    for k, p in enumerate(suj_train):
        for etq, marca in [(0, "o"), (1, "^")]:
            i = (suj == p) & (act == etq)
            axes[1].scatter(PC[i, 0], PC[i, 1], color=colores[k], marker=marca,
                            s=22 if etq == 0 else 45, alpha=0.5 if etq == 0 else 0.95,
                            edgecolor="k" if etq else "none", linewidth=0.5,
                            label=p if etq == 0 else None)
    axes[1].set(title="PCA 2D de la MDE (○ rest, ▲ valsalva; color = sujeto)",
                xlabel="PC1 (u. a.)", ylabel="PC2 (u. a.)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)
    plt.tight_layout(); guardar("mde_correlacion_pca.png")

    eta = pd.DataFrame({f"PC{k+1}": {"η² sujeto": eta2(PC[:, k], suj),
                                     "η² clase": eta2(PC[:, k], act)} for k in range(3)})
    print(eta.round(3))
    eta.to_csv(SALIDA / "eta2_pca_mde.csv")


if __name__ == "__main__":
    main()
