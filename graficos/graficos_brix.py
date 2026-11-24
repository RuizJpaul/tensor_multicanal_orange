"""Gráficos de informe para la Rama 3: Mediciones Físico-Químicas °Brix (MLP).

Genera en graficos/graficos_brix_mlp/:
  01_distribucion_brix_por_punto.png
  02_correlacion_y_perfil_brix.png
  03_curvas_entrenamiento_brix.png
  04_matrices_confusion_brix.png
  05_curvas_roc_pr_brix.png
  06_distribucion_probabilidades_brix.png
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

CARPETA_SALIDA = GRAFICOS / "graficos_brix_mlp"


def graficar_distribucion_brix(X_brix: np.ndarray, cols: list[str], y: np.ndarray) -> None:
  media_muestra = X_brix.mean(axis=1, keepdims=True)
  std_muestra = X_brix.std(axis=1, keepdims=True)
  datos_ext = np.hstack([X_brix, media_muestra, std_muestra])
  nombres = [f"Punto {c}" for c in cols] + ["Promedio (P1–P4)", "Desv. Est. Interna"]

  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  fig.suptitle("Distribución de °Brix por Punto de Medición y Estadísticos Internos según Clase", fontweight="bold", fontsize=13)
  axes = axes.ravel()

  rng = np.random.default_rng(42)
  for i, titulo in enumerate(nombres):
    ax = axes[i]
    d0 = datos_ext[y == 0, i]
    d1 = datos_ext[y == 1, i]
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
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("°Brix")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

  plt.tight_layout(rect=[0, 0, 1, 0.94])
  guardar_figura(fig, CARPETA_SALIDA / "01_distribucion_brix_por_punto.png")


def graficar_correlacion_y_perfil_brix(X_brix: np.ndarray, cols: list[str], y: np.ndarray, ids: np.ndarray) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), gridspec_kw={"width_ratios": [1, 1.8]})
  fig.suptitle("Correlación entre Puntos °Brix y Perfil por Muestra (M1–M100)", fontweight="bold", fontsize=13)

  # 1. Matriz de correlación P1..P4
  ax = axes[0]
  corr = np.corrcoef(X_brix, rowvar=False)
  im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
  ax.set_xticks(range(len(cols)))
  ax.set_yticks(range(len(cols)))
  ax.set_xticklabels(cols)
  ax.set_yticklabels(cols)
  ax.set_title("Correlación de Pearson (P1–P4)", fontweight="bold")
  for i in range(len(cols)):
    for j in range(len(cols)):
      c_txt = "white" if abs(corr[i, j]) > 0.6 else "black"
      ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=c_txt, fontweight="bold", fontsize=10)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

  # 2. Perfil de media ± dispersión de Brix por muestra M1..M100
  ax = axes[1]
  idx = np.arange(len(X_brix))
  m = X_brix.mean(axis=1)
  s = X_brix.std(axis=1)
  ax.plot(idx, m, color=COLOR_SANA, lw=1.5, marker="o", markersize=3.5, label="°Brix medio (Sana)")
  ax.fill_between(idx, m - s, m + s, color=COLOR_SANA, alpha=0.16, label="±1 DE entre P1–P4")
  idx_g = np.where(y == 1)[0]
  ax.scatter(idx_g, m[idx_g], color=COLOR_GRAN, s=42, zorder=5, label="Clase 1 (Granulada)")
  for g in idx_g:
    ax.annotate(ids[g], xy=(g, m[g]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=COLOR_GRAN, fontweight="bold")

  paso = max(1, len(X_brix) // 20)
  ax.set_xticks(idx[::paso])
  ax.set_xticklabels(ids[::paso], rotation=45, ha="right", fontsize=8.5)
  ax.set_title("Perfil de °Brix Medio y Dispersión Interna por Muestra", fontweight="bold")
  ax.set_xlabel("Muestra")
  ax.set_ylabel("°Brix")
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=8.5, loc="upper right")

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, CARPETA_SALIDA / "02_correlacion_y_perfil_brix.png")


def main() -> None:
  print(">>> Generando gráficos para Rama °Brix MLP...")
  ids, y = ud.cargar_etiquetas()
  X_brix, cols = ud.cargar_brix(ids)

  graficar_distribucion_brix(X_brix, cols, y)
  graficar_correlacion_y_perfil_brix(X_brix, cols, y, ids)
  graficar_curvas_entrenamiento("brix", "Rama °Brix MLP (P1–P4)", CARPETA_SALIDA, "03_curvas_entrenamiento_brix.png")
  graficar_matrices_confusion("brix", "Rama °Brix MLP (P1–P4)", CARPETA_SALIDA, "04_matrices_confusion_brix.png")
  graficar_curvas_roc_y_pr("brix", "Rama °Brix MLP (P1–P4)", CARPETA_SALIDA, "05_curvas_roc_pr_brix.png")
  graficar_distribucion_probabilidades("brix", "Rama °Brix MLP (P1–P4)", CARPETA_SALIDA, "06_distribucion_probabilidades_brix.png")
  print(f"Listo. 6 gráficos guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
  main()
