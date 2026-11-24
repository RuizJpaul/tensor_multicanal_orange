"""Gráficos de informe para la Rama 2: Visión Multicanal CNN (RGB + UV + NIR).

Genera en graficos/graficos_fotos_cnn/:
  01_ejemplos_canales_sana_vs_granulada.png
  02_curvas_entrenamiento_cnn.png
  03_matrices_confusion_cnn.png
  04_curvas_roc_pr_cnn.png
  05_distribucion_probabilidades_cnn.png
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
  sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np

from graficos.utilidades_graficos import (
    GRAFICOS,
    graficar_curvas_entrenamiento,
    graficar_curvas_roc_y_pr,
    graficar_distribucion_probabilidades,
    graficar_matrices_confusion,
    guardar_figura,
)
import utilidades_datos as ud

CARPETA_SALIDA = GRAFICOS / "graficos_fotos_cnn"


def graficar_ejemplos_canales(X_img: np.ndarray, y: np.ndarray, ids: np.ndarray) -> None:
  idx_sanas = np.where(y == 0)[0][:2]
  idx_gran = np.where(y == 1)[0][:2]
  seleccion = [(idx_sanas[0], "Sana (0)"), (idx_sanas[1], "Sana (0)"),
               (idx_gran[0], "Granulada (1)"), (idx_gran[1], "Granulada (1)")]

  fig, axes = plt.subplots(4, 4, figsize=(13, 12))
  fig.suptitle("Representación del Tensor Multicanal (224×224×3) — Muestras Sanas vs. Granuladas", fontweight="bold", fontsize=13.5)
  titulos_col = ["Canal 0: RGB (Gris ROI)", "Canal 1: Ultravioleta (UV)", "Canal 2: Infrarrojo Cercano (NIR)", "Falso Color (Tensor 3 Canales)"]

  for fila, (idx_m, etq_txt) in enumerate(seleccion):
    tensor = X_img[idx_m]
    for col in range(3):
      ax = axes[fila, col]
      ax.imshow(tensor[:, :, col], cmap="inferno" if col > 0 else "gray", vmin=0, vmax=1)
      if fila == 0:
        ax.set_title(titulos_col[col], fontweight="bold", fontsize=10)
      if col == 0:
        ax.set_ylabel(f"{ids[idx_m]}\n{etq_txt}", fontweight="bold", fontsize=10.5)
      ax.set_xticks([])
      ax.set_yticks([])

    ax = axes[fila, 3]
    ax.imshow(np.clip(tensor, 0, 1))
    if fila == 0:
      ax.set_title(titulos_col[3], fontweight="bold", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

  plt.tight_layout(rect=[0, 0, 1, 0.95])
  guardar_figura(fig, CARPETA_SALIDA / "01_ejemplos_canales_sana_vs_granulada.png")


def main() -> None:
  print(">>> Generando gráficos para Rama CNN Multicanal...")
  ids, y = ud.cargar_etiquetas()
  X_img = ud.cargar_imagenes(ids, y_ref=y)

  graficar_ejemplos_canales(X_img, y, ids)
  graficar_curvas_entrenamiento("cnn", "Rama CNN Multicanal (RGB+UV+NIR)", CARPETA_SALIDA, "02_curvas_entrenamiento_cnn.png")
  graficar_matrices_confusion("cnn", "Rama CNN Multicanal (RGB+UV+NIR)", CARPETA_SALIDA, "03_matrices_confusion_cnn.png")
  graficar_curvas_roc_y_pr("cnn", "Rama CNN Multicanal (RGB+UV+NIR)", CARPETA_SALIDA, "04_curvas_roc_pr_cnn.png")
  graficar_distribucion_probabilidades("cnn", "Rama CNN Multicanal (RGB+UV+NIR)", CARPETA_SALIDA, "05_distribucion_probabilidades_cnn.png")
  print(f"Listo. 5 gráficos guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
  main()
