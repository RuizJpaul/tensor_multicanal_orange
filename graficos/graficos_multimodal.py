"""Gráficos de informe para el Modelo de Fusión Multimodal y Comparativa Global.

Genera en graficos/graficos_multimodal/:
  01_curvas_entrenamiento_multimodal.png
  02_matrices_confusion_multimodal.png
  03_curvas_roc_pr_multimodal.png
  04_distribucion_probabilidades_multimodal.png
  05_comparativa_global_modelos_cv.png
  06_curvas_roc_comparadas_todos_modelos.png
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
  sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from graficos.utilidades_graficos import (
    GRAFICOS,
    RESULTADOS,
    graficar_curvas_entrenamiento,
    graficar_curvas_roc_y_pr,
    graficar_distribucion_probabilidades,
    graficar_matrices_confusion,
    guardar_figura,
)

CARPETA_SALIDA = GRAFICOS / "graficos_multimodal"

NOMBRES_MODELOS = {
    "multimodal": "Fusión Multimodal",
    "brix": "Rama °Brix MLP",
    "nir": "Rama NIR MLP",
    "cnn": "Rama CNN Multicanal",
    "logreg_nir": "Baseline LogReg NIR",
    "logreg_brix": "Baseline LogReg °Brix",
}
COLORES_MODELOS = {
    "multimodal": "#d95f02",
    "brix": "#1b9e77",
    "nir": "#7570b3",
    "cnn": "#2b5c8f",
    "logreg_nir": "#8c8c8c",
    "logreg_brix": "#b5b5b5",
}


def graficar_comparativa_global_cv() -> None:
  df = pd.read_csv(RESULTADOS / "cv_resumen.csv")
  df_val = df[df["tipo_umbral"] == "validacion"].copy()
  orden = ["multimodal", "brix", "nir", "cnn", "logreg_nir", "logreg_brix"]
  df_val["orden"] = df_val["rama"].map({r: i for i, r in enumerate(orden)})
  df_val = df_val.sort_values("orden").reset_index(drop=True)

  metricas = [
      ("roc_auc", "ROC-AUC"),
      ("f1_granulada", "F1-Score (Granulada)"),
      ("recall_granulada", "Sensibilidad / Recall"),
      ("balanced_accuracy", "Balanced Accuracy"),
  ]

  fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
  fig.suptitle(
      "Comparativa Global de Modelos — Validación Cruzada Estratificada Repetida (5 Folds × 3 Reps, n=100)",
      fontweight="bold", fontsize=13.5,
  )
  axes = axes.ravel()

  etiquetas = [NOMBRES_MODELOS.get(r, r) for r in df_val["rama"]]
  colores = [COLORES_MODELOS.get(r, "#333333") for r in df_val["rama"]]
  x = np.arange(len(df_val))

  for idx_m, (col_m, titulo_m) in enumerate(metricas):
    ax = axes[idx_m]
    medias = df_val[f"{col_m}_mean"].to_numpy()
    stds = df_val[f"{col_m}_std"].to_numpy()
    barras = ax.bar(x, medias, yerr=stds, capsize=5, color=colores, edgecolor="black", width=0.58, alpha=0.9)
    if col_m in ("roc_auc", "balanced_accuracy"):
      ax.axhline(0.5, color="gray", linestyle="--", lw=1.2, label="Nivel de azar (0.50)")
      ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(titulo_m, fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for b, m, s in zip(barras, medias, stds):
      ax.text(
          b.get_x() + b.get_width() / 2, min(0.98, m + s + 0.025),
          f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
      )

  plt.tight_layout(rect=[0, 0, 1, 0.94])
  guardar_figura(fig, CARPETA_SALIDA / "05_comparativa_global_modelos_cv.png")


def graficar_curvas_roc_comparadas() -> None:
  ramas = ["multimodal", "brix", "nir", "cnn"]
  fpr_grid = np.linspace(0, 1, 101)

  fig, ax = plt.subplots(figsize=(8.2, 6.6))
  for rama in ramas:
    pred_cv = pd.read_csv(RESULTADOS / f"cv_predicciones_{rama}.csv")
    tprs, aucs = [], []
    for _, df_rep in pred_cv.groupby("repeticion"):
      fpr, tpr, _ = roc_curve(df_rep["y_real"], df_rep["prob_granulada"])
      aucs.append(auc(fpr, tpr))
      t_interp = np.interp(fpr_grid, fpr, tpr)
      t_interp[0] = 0.0
      tprs.append(t_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    lw = 2.8 if rama == "multimodal" else 2.0
    ax.plot(
        fpr_grid, mean_tpr, lw=lw, color=COLORES_MODELOS[rama],
        label=f"{NOMBRES_MODELOS[rama]} (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})",
    )
    if rama == "multimodal":
      std_tpr = np.std(tprs, axis=0)
      ax.fill_between(
          fpr_grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
          color=COLORES_MODELOS[rama], alpha=0.15, label="±1 DE Fusión Multimodal",
      )

  ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.4, label="Azar (AUC = 0.500)")
  ax.set_title("Curvas ROC Medias Comparadas (Validación Cruzada 5×3, n=100)", fontweight="bold", fontsize=12.5)
  ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=10.5)
  ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontsize=10.5)
  ax.set_xlim(-0.02, 1.02)
  ax.set_ylim(-0.02, 1.05)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(loc="lower right", fontsize=9.5)

  plt.tight_layout()
  guardar_figura(fig, CARPETA_SALIDA / "06_curvas_roc_comparadas_todos_modelos.png")


def main() -> None:
  print(">>> Generando gráficos para Modelo de Fusión Multimodal y Comparativa Global...")
  graficar_curvas_entrenamiento("multimodal", "Fusión Multimodal (Imagen + NIR + °Brix)", CARPETA_SALIDA, "01_curvas_entrenamiento_multimodal.png")
  graficar_matrices_confusion("multimodal", "Fusión Multimodal (Imagen + NIR + °Brix)", CARPETA_SALIDA, "02_matrices_confusion_multimodal.png")
  graficar_curvas_roc_y_pr("multimodal", "Fusión Multimodal (Imagen + NIR + °Brix)", CARPETA_SALIDA, "03_curvas_roc_pr_multimodal.png")
  graficar_distribucion_probabilidades("multimodal", "Fusión Multimodal (Imagen + NIR + °Brix)", CARPETA_SALIDA, "04_distribucion_probabilidades_multimodal.png")
  graficar_comparativa_global_cv()
  graficar_curvas_roc_comparadas()
  print(f"Listo. 6 gráficos guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
  main()
