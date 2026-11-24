"""Funciones compartidas de visualización académica para los 4 modelos.

Genera gráficos de alta resolución (DPI=200) listos para el informe Capstone
a partir de los archivos en resultados/ y datos/:
  - Curvas de aprendizaje (Loss, ROC-AUC y Accuracy en Train vs. Validación).
  - Matrices de confusión (Test fijo n=15 y Validación Cruzada OOF n=100).
  - Curvas ROC (con banda de ±1 desv. est. entre repeticiones CV) y Precision-Recall.
  - Distribución de probabilidades predichas por clase y umbrales de decisión.
"""
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
GRAFICOS = RAIZ / "graficos"

COLOR_SANA = "#2b5c8f"
COLOR_GRAN = "#d95f02"
COLOR_TRAIN = "#1f77b4"
COLOR_VAL = "#ff7f0e"


def guardar_figura(fig: plt.Figure, ruta_archivo: Path) -> None:
  ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
  for intento in range(3):
    try:
      fig.savefig(str(ruta_archivo), dpi=200, bbox_inches="tight")
      break
    except OSError:
      if intento == 2:
        raise
      time.sleep(0.3)
  plt.close(fig)
  print(f"  [OK] Guardada: {ruta_archivo.relative_to(RAIZ)}")


def cargar_artefactos_rama(rama: str):
  with open(RESULTADOS / f"metricas_{rama}.json", encoding="utf-8") as f:
    metricas = json.load(f)
  historial = pd.read_csv(RESULTADOS / f"historial_{rama}.csv")
  pred_test = pd.read_csv(RESULTADOS / f"predicciones_test_{rama}.csv")
  pred_cv = pd.read_csv(RESULTADOS / f"cv_predicciones_{rama}.csv")
  return metricas, historial, pred_test, pred_cv


def graficar_curvas_entrenamiento(rama: str, nombre_modelo: str, carpeta_salida: Path, nombre_archivo: str) -> None:
  _, hist, _, _ = cargar_artefactos_rama(rama)
  epocas = hist["epoca"] + 1
  mejor_idx = int(hist["val_loss"].idxmin())
  mejor_epoca = int(epocas.iloc[mejor_idx])

  fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
  fig.suptitle(f"Dinámica de Entrenamiento y Validación — {nombre_modelo}", fontweight="bold", fontsize=13)

  # 1. Pérdida (Binary Crossentropy)
  ax = axes[0]
  ax.plot(epocas, hist["loss"], label="Train Loss", color=COLOR_TRAIN, lw=2)
  ax.plot(epocas, hist["val_loss"], label="Val Loss", color=COLOR_VAL, lw=2)
  ax.axvline(mejor_epoca, color="#2ca02c", linestyle="--", lw=1.4, label=f"Mejor época ({mejor_epoca})")
  ax.scatter([mejor_epoca], [hist["val_loss"].iloc[mejor_idx]], color="#2ca02c", zorder=5, s=45)
  ax.set_title("Pérdida (Binary Cross-Entropy)", fontweight="bold")
  ax.set_xlabel("Época")
  ax.set_ylabel("Loss")
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=8.5)

  # 2. ROC-AUC
  ax = axes[1]
  ax.plot(epocas, hist["auc"], label="Train ROC-AUC", color=COLOR_TRAIN, lw=2)
  ax.plot(epocas, hist["val_auc"], label="Val ROC-AUC", color=COLOR_VAL, lw=2)
  ax.axvline(mejor_epoca, color="#2ca02c", linestyle="--", lw=1.4)
  ax.axhline(0.5, color="gray", linestyle=":", lw=1.2, label="Azar (0.50)")
  ax.set_title("Evolución de ROC-AUC", fontweight="bold")
  ax.set_xlabel("Época")
  ax.set_ylabel("ROC-AUC")
  ax.set_ylim(0.0, 1.05)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=8.5)

  # 3. Exactitud (Accuracy)
  ax = axes[2]
  ax.plot(epocas, hist["accuracy"], label="Train Accuracy", color=COLOR_TRAIN, lw=2)
  ax.plot(epocas, hist["val_accuracy"], label="Val Accuracy", color=COLOR_VAL, lw=2)
  ax.axvline(mejor_epoca, color="#2ca02c", linestyle="--", lw=1.4)
  ax.set_title("Evolución de Exactitud (Accuracy)", fontweight="bold")
  ax.set_xlabel("Época")
  ax.set_ylabel("Accuracy")
  ax.set_ylim(0.0, 1.05)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=8.5)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, carpeta_salida / nombre_archivo)


def _dibujar_matriz(ax, cm: np.ndarray, titulo: str, subtitulo: str) -> None:
  cm = np.asarray(cm, dtype=int)
  totales_fila = cm.sum(axis=1, keepdims=True)
  pct = np.divide(cm, totales_fila, out=np.zeros_like(cm, dtype=float), where=totales_fila != 0) * 100

  im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
  ax.set_xticks([0, 1])
  ax.set_yticks([0, 1])
  ax.set_xticklabels(["Pred: Sana (0)", "Pred: Granulada (1)"], fontsize=9.5)
  ax.set_yticklabels(["Real: Sana (0)", "Real: Granulada (1)"], fontsize=9.5)
  ax.set_title(f"{titulo}\n{subtitulo}", fontsize=10, fontweight="bold")

  for i in range(2):
    for j in range(2):
      color_txt = "white" if pct[i, j] > 55 else "black"
      ax.text(
          j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
          ha="center", va="center", color=color_txt, fontsize=11, fontweight="bold",
      )
  return im


def graficar_matrices_confusion(rama: str, nombre_modelo: str, carpeta_salida: Path, nombre_archivo: str) -> None:
  metricas, _, _, pred_cv = cargar_artefactos_rama(rama)
  u_val = metricas["umbral_validacion"]

  cm_test_05 = np.array(metricas["test_umbral_050"]["matriz_confusion"])
  cm_test_val = np.array(metricas["test_umbral_validacion"]["matriz_confusion"])

  # OOF promedio sumado o sobre las 3 repeticiones (n=300 evaluaciones = 100 muestras x 3 reps)
  y_cv = pred_cv["y_real"].to_numpy()
  p_cv = pred_cv["prob_granulada"].to_numpy()
  u_cv = pred_cv["umbral_validacion"].to_numpy()
  cm_cv_05 = confusion_matrix(y_cv, (p_cv >= 0.5).astype(int), labels=[0, 1])
  cm_cv_val = confusion_matrix(y_cv, (p_cv >= u_cv).astype(int), labels=[0, 1])

  fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
  fig.suptitle(f"Matrices de Confusión — {nombre_modelo}", fontweight="bold", fontsize=13.5)

  _dibujar_matriz(
      axes[0, 0], cm_test_05,
      "Partición Test Fija (n=15) — Umbral 0.50",
      f"Acc={metricas['test_umbral_050']['accuracy']:.3f} | F1={metricas['test_umbral_050']['f1_granulada']:.3f}",
  )
  _dibujar_matriz(
      axes[0, 1], cm_test_val,
      f"Partición Test Fija (n=15) — Umbral Val ({u_val:.3f})",
      f"Acc={metricas['test_umbral_validacion']['accuracy']:.3f} | F1={metricas['test_umbral_validacion']['f1_granulada']:.3f}",
  )
  _dibujar_matriz(
      axes[1, 0], cm_cv_05,
      "Validación Cruzada 5×3 OOF (300 eval.) — Umbral 0.50",
      f"Recall Gran.={cm_cv_05[1,1]/max(1,cm_cv_05[1].sum()):.1%} | Especif.={cm_cv_05[0,0]/max(1,cm_cv_05[0].sum()):.1%}",
  )
  _dibujar_matriz(
      axes[1, 1], cm_cv_val,
      "Validación Cruzada 5×3 OOF (300 eval.) — Umbral Val",
      f"Recall Gran.={cm_cv_val[1,1]/max(1,cm_cv_val[1].sum()):.1%} | Especif.={cm_cv_val[0,0]/max(1,cm_cv_val[0].sum()):.1%}",
  )

  plt.tight_layout(rect=[0, 0, 1, 0.94])
  guardar_figura(fig, carpeta_salida / nombre_archivo)


def graficar_curvas_roc_y_pr(rama: str, nombre_modelo: str, carpeta_salida: Path, nombre_archivo: str) -> None:
  _, _, pred_test, pred_cv = cargar_artefactos_rama(rama)

  fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
  fig.suptitle(f"Curvas ROC y Precision-Recall — {nombre_modelo}", fontweight="bold", fontsize=13)

  # 1. Curva ROC con media ± 1 std entre repeticiones CV + Test fijo
  ax = axes[0]
  fpr_grid = np.linspace(0, 1, 101)
  tprs, aucs = [], []
  for rep, df_rep in pred_cv.groupby("repeticion"):
    fpr, tpr, _ = roc_curve(df_rep["y_real"], df_rep["prob_granulada"])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    ax.plot(fpr, tpr, lw=1.1, alpha=0.35, label=f"CV Rep {rep + 1} (AUC = {roc_auc:.3f})")

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  std_tpr = np.std(tprs, axis=0)
  ax.plot(
      fpr_grid, mean_tpr, color=COLOR_GRAN, lw=2.5,
      label=f"Media CV 5×3 (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})",
  )
  ax.fill_between(
      fpr_grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
      color=COLOR_GRAN, alpha=0.18, label="±1 Desv. Est. (CV)",
  )

  # Test fijo
  fpr_te, tpr_te, _ = roc_curve(pred_test["y_real"], pred_test["prob_granulada"])
  ax.plot(
      fpr_te, tpr_te, color=COLOR_SANA, linestyle="--", lw=1.8,
      label=f"Test Fijo n=15 (AUC = {auc(fpr_te, tpr_te):.3f})",
  )
  ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.2, label="Clasificador aleatorio (0.500)")
  ax.set_title("Curva ROC (Receiver Operating Characteristic)", fontweight="bold")
  ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
  ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)")
  ax.set_xlim(-0.02, 1.02)
  ax.set_ylim(-0.02, 1.05)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(loc="lower right", fontsize=8.5)

  # 2. Curva Precision-Recall (CV OOF y Test)
  ax = axes[1]
  aps = []
  for rep, df_rep in pred_cv.groupby("repeticion"):
    prec, rec, _ = precision_recall_curve(df_rep["y_real"], df_rep["prob_granulada"])
    ap = average_precision_score(df_rep["y_real"], df_rep["prob_granulada"])
    aps.append(ap)
    ax.plot(rec, prec, lw=1.2, alpha=0.45, label=f"CV Rep {rep + 1} (AP = {ap:.3f})")

  prec_all, rec_all, _ = precision_recall_curve(pred_cv["y_real"], pred_cv["prob_granulada"])
  ax.plot(
      rec_all, prec_all, color=COLOR_GRAN, lw=2.4,
      label=f"Global CV OOF (AP medio = {np.mean(aps):.3f} ± {np.std(aps):.3f})",
  )
  prevalencia = pred_cv["y_real"].mean()
  ax.axhline(prevalencia, color="gray", linestyle=":", lw=1.3, label=f"Prevalencia base ({prevalencia:.2f})")
  ax.set_title("Curva Precision-Recall (Clase Granulada = 1)", fontweight="bold")
  ax.set_xlabel("Sensibilidad / Recall (Granulada)")
  ax.set_ylabel("Precisión (Granulada)")
  ax.set_xlim(-0.02, 1.02)
  ax.set_ylim(0.0, 1.05)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(loc="upper right", fontsize=8.5)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, carpeta_salida / nombre_archivo)


def graficar_distribucion_probabilidades(rama: str, nombre_modelo: str, carpeta_salida: Path, nombre_archivo: str) -> None:
  metricas, _, pred_test, pred_cv = cargar_artefactos_rama(rama)
  u_val = metricas["umbral_validacion"]

  fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
  fig.suptitle(f"Distribución de Probabilidades Predichas — {nombre_modelo}", fontweight="bold", fontsize=13)

  # 1. Predicciones OOF promediadas por muestra (n=100)
  ax = axes[0]
  oof_media = pred_cv.groupby(["ID", "y_real"], as_index=False)["prob_granulada"].mean()
  p0 = oof_media[oof_media["y_real"] == 0]["prob_granulada"].to_numpy()
  p1 = oof_media[oof_media["y_real"] == 1]["prob_granulada"].to_numpy()
  u_cv_medio = pred_cv["umbral_validacion"].mean()

  bp = ax.boxplot(
      [p0, p1],
      tick_labels=[f"Sana (0)\nn = {len(p0)}", f"Granulada (1)\nn = {len(p1)}"],
      patch_artist=True,
      widths=0.45,
  )
  bp["boxes"][0].set_facecolor("#a6bddb")
  bp["boxes"][1].set_facecolor("#fdbb84")

  rng = np.random.default_rng(42)
  ax.scatter(1 + rng.normal(0, 0.05, len(p0)), p0, color=COLOR_SANA, alpha=0.55, s=24, label="Muestras Sanas")
  ax.scatter(2 + rng.normal(0, 0.05, len(p1)), p1, color=COLOR_GRAN, alpha=0.9, s=36, zorder=4, label="Muestras Granuladas")
  ax.axhline(0.5, color="black", linestyle="--", lw=1.3, label="Umbral fijo (0.50)")
  ax.axhline(u_cv_medio, color="#2ca02c", linestyle="-.", lw=1.5, label=f"Umbral medio CV ({u_cv_medio:.3f})")
  ax.set_title("Probabilidad OOF Media por Muestra (Validación Cruzada, n=100)", fontweight="bold")
  ax.set_ylabel("Probabilidad predicha de granulación")
  ax.set_ylim(-0.05, 1.05)
  ax.grid(axis="y", alpha=0.3, linestyle="--")
  ax.legend(fontsize=8.5, loc="upper left")

  # 2. Muestra por muestra en el conjunto de Test fijo (n=15)
  ax = axes[1]
  df_te = pred_test.sort_values("prob_granulada", ascending=False).reset_index(drop=True)
  colores = [COLOR_GRAN if y == 1 else COLOR_SANA for y in df_te["y_real"]]
  barras = ax.bar(range(len(df_te)), df_te["prob_granulada"], color=colores, edgecolor="black", alpha=0.88)
  ax.axhline(0.5, color="black", linestyle="--", lw=1.3, label="Umbral fijo (0.50)")
  ax.axhline(u_val, color="#2ca02c", linestyle="-.", lw=1.5, label=f"Umbral validación ({u_val:.3f})")
  ax.set_xticks(range(len(df_te)))
  ax.set_xticklabels(df_te["ID"], rotation=45, ha="right", fontsize=9)
  ax.set_ylim(0, 1.08)
  ax.set_title("Probabilidad por Muestra en Partición Test Fija (n=15)", fontweight="bold")
  ax.set_ylabel("Probabilidad predicha de granulación")
  ax.grid(axis="y", alpha=0.3, linestyle="--")

  # Leyenda manual para barras + umbrales
  from matplotlib.patches import Patch
  handles, labels = ax.get_legend_handles_labels()
  handles = [
      Patch(facecolor=COLOR_SANA, edgecolor="black", label="Real: Sana (0)"),
      Patch(facecolor=COLOR_GRAN, edgecolor="black", label="Real: Granulada (1)"),
  ] + handles
  ax.legend(handles=handles, fontsize=8.5, loc="upper right")

  for b, p in zip(barras, df_te["prob_granulada"]):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=7.5)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, carpeta_salida / nombre_archivo)
