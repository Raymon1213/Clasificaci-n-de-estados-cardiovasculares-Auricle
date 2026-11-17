#!/usr/bin/env python3
"""
Ejecuta el flujo completo en orden (cada paso es un script independiente):

  1. eda.py              EDA, anotaciones, ventanas válidas
  2. mdef.py             partición 9/3 por sujeto + MDE + estandarización con Train
  3. exploracion_mde.py  gráficos exploratorios (solo Train)
  4. dimr.py             SLS (RMSprop + RLS) + PCA-SVD, ajustados con Train
  5. train.py            Softmax ponderado con RMSprop (Train)
  6. test.py             evaluación única en Test
  7. experimento_alt.py  rama experimental (LOSO-Train + Test)

Uso:  python run_all.py                 # todo
      python run_all.py --desde dimr    # reanuda desde un paso (nombre sin .py)
      python run_all.py --sin-alt       # omite el experimento adicional
"""
import argparse
import subprocess
import sys
from pathlib import Path

PASOS = ["eda", "mdef", "exploracion_mde", "dimr", "train", "test", "experimento_alt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", choices=PASOS, default=PASOS[0])
    ap.add_argument("--sin-alt", action="store_true")
    args = ap.parse_args()

    carpeta = Path(__file__).resolve().parent
    pasos = PASOS[PASOS.index(args.desde):]
    if args.sin_alt:
        pasos = [p for p in pasos if p != "experimento_alt"]
    for p in pasos:
        print(f"\n{'=' * 70}\n>>> {p}.py\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(carpeta / f"{p}.py")], check=True, cwd=carpeta)
    print("\nFlujo completo. Resultados en experimento_integrado/")


if __name__ == "__main__":
    main()
