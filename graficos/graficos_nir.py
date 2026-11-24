"""Gráficos de informe para la Rama 1: Espectroscopía NIR (Sensor AS7263, MLP).

Genera en graficos/graficos_nir_mlp/:
  01_firma_espectral_nir_por_clase.png
  02_boxplots_bandas_nir.png
  03_curvas_entrenamiento_nir.png
  04_matrices_confusion_nir.png
  05_curvas_roc_pr_nir.png
  06_distribucion_probabilidades_nir.png
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
  sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np

from graficos.utilidades_graficos import (
    COLOR_GRAN,
    COLOR_SANA,
    GRAFICOS,
    graficar_curvas_entrenamiento,
    graficar_curvas_roc_y_pr,
    graficar_distribucion_probabilidades,
    graficar_matrices_confusion,
    guardar_figura,
)
import utilidades_datos as ud

CARPETA_SALIDA = GRAFICOS / "graficos_nir_mlp"
LONGITUDES_ONDA = {"R": "610 nm", "S": "680 nm", "T": "730 nm", "U": "760 nm", "V": "810 nm", "W": "860 nm"}
CANALES = ["R", "S", "T", "U", "V", "W"]


def graficar_firma_espectral(X_nir: np.ndarray, cols: list[str], y: np.ndarray) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  fig.suptitle("Caracterización Espectral NIR AS7263 por Clase (Sana vs. Granulada)", fontweight="bold", fontsize=13)

  # 1. Perfil completo sobre las 24 variables (P1..P4 x R..W)
  ax = axes[0]
  x_idx = np.arange(len(cols))
  for clase, nombre, color in ((0, f"Sana (0) [n={int((y==0).sum())}]", COLOR_SANA),
                               (1, f"Granulada (1) [n={int((y==1).sum())}]", COLOR_GRAN)):
    sub = X_nir[y == clase]
    media = sub.mean(axis=0)
    std = sub.std(axis=0)
    ax.plot(x_idx, media, marker="o", markersize=4, lw=2, color=color, label=nombre)
    ax.fill_between(x_idx, media - std, media + std, color=color, alpha=0.16)

  for sep in (5.5, 11.5, 17.5):
    ax.axvline(sep, color="gray", linestyle="--", lw=1, alpha=0.6)
  ax.set_xticks(x_idx)
  ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
  ax.set_title("Perfil de las 24 Variables Espectrales (Puntos P1 a P4)", fontweight="bold")
  ax.set_ylabel("Reflectancia / Intensidad AS7263")
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=9)

  # 2. Firma espectral promediando los 4 puntos ecuatoriales por longitud de onda (6 bandas)
  ax = axes[1]
  # X_nir tiene orden R_P1..W_P1, R_P2..W_P2, R_P3..W_P3, R_P4..W_P4 -> reshape (N, 4, 6)
  X_bandas = X_nir.reshape(-1, 4, 6).mean(axis=1)
  etiquetas_onda = [f"{c}\n({LONGITUDES_ONDA[c]})" for c in CANALES]
  b_idx = np.arange(len(CANALES))

  for clase, nombre, color in ((0, "Media Sana (0) ± 1 DE", COLOR_SANA),
                               (1, "Media Granulada (1) ± 1 DE", COLOR_GRAN)):
    sub = X_bandas[y == clase]
    media = sub.mean(axis=0)
    std = sub.std(axis=0)
    ax.plot(b_idx, media, marker="s", markersize=6, lw=2.3, color=color, label=nombre)
    ax.fill_between(b_idx, media - std, media + std, color=color, alpha=0.18)

  ax.set_xticks(b_idx)
  ax.set_xticklabels(etiquetas_onda, fontsize=9.5)
  ax.set_title("Firma Espectral Media por Longitud de Onda (610–860 nm)", fontweight="bold")
  ax.set_xlabel("Canal Espectral AS7263 (Longitud de onda central)")
  ax.set_ylabel("Intensidad media (P1–P4)")
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=9)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, CARPETA_SALIDA / "01_firma_espectral_nir_por_clase.png")


def graficar_boxplots_bandas(X_nir: np.ndarray, y: np.ndarray) -> None:
  X_bandas = X_nir.reshape(-1, 4, 6).mean(axis=1)
  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  fig.suptitle("Distribución por Banda Espectral NIR (Promedio P1–P4) según Clase", fontweight="bold", fontsize=13)
  axes = axes.ravel()

  rng = np.random.default_rng(42)
  for i, canal in enumerate(CANALES):
    ax = axes[i]
    d0 = X_bandas[y == 0, i]
    d1 = X_bandas[y == 1, i]
    bp = ax.boxplot(
        [d0, d1],
        tick_labels=[f"Sana (0)\nn={len(d0)}", f"Granulada (1)\nn={len(d1)}"],
        patch_artist=True,
        widths=0.48,
    )
    bp["boxes"][0].set_facecolor("#a6bddb")
    bp["boxes"][1].set_facecolor("#fdbb84")
    ax.scatter(1 + rng.normal(0, 0.05, len(d0)), d0, color=COLOR_SANA, alpha=0.45, s=18)
    ax.scatter(2 + rng.normal(0, 0.05, len(d1)), d1, color=COLOR_GRAN, alpha=0.85, s=26, zorder=4)
    ax.set_title(f"Canal {canal} ({LONGITUDES_ONDA[canal]})", fontweight="bold")
    ax.set_ylabel("Intensidad NIR")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

  plt.tight_layout(rect=[0, 0, 1, 0.94])
  guardar_figura(fig, CARPETA_SALIDA / "02_boxplots_bandas_nir.png")


def main() -> None:
  print(">>> Generando gráficos para Rama NIR MLP...")
  ids, y = ud.cargar_etiquetas()
  X_nir, cols = ud.cargar_nir(ids)

  graficar_firma_espectral(X_nir, cols, y)
  graficar_boxplots_bandas(X_nir, y)
  graficar_curvas_entrenamiento("nir", "Rama NIR MLP (AS7263)", CARPETA_SALIDA, "03_curvas_entrenamiento_nir.png")
  graficar_matrices_confusion("nir", "Rama NIR MLP (AS7263)", CARPETA_SALIDA, "04_matrices_confusion_nir.png")
  graficar_curvas_roc_y_pr("nir", "Rama NIR MLP (AS7263)", CARPETA_SALIDA, "05_curvas_roc_pr_nir.png")
  graficar_distribucion_probabilidades("nir", "Rama NIR MLP (AS7263)", CARPETA_SALIDA, "06_distribucion_probabilidades_nir.png")
  print(f"Listo. 6 gráficos guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
  main()
