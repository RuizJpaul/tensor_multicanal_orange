import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Ajusta la ruta si ejecutas el script desde otra carpeta
RUTA_OOF = (
    "resultados_cv/oof_probabilidades.npz"
    if os.path.exists("resultados_cv/oof_probabilidades.npz")
    else "oof_probabilidades.npz"
)
CARPETA_SALIDA = "resultados_cv/figuras"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

data = np.load(RUTA_OOF)
y = data["y"]

modelos = ["nir", "brix", "cnn", "multimodal"]
etiquetas = {
    "nir": "NIR (MLP)",
    "brix": "Brix (MLP)",
    "cnn": "CNN Multicanal",
    "multimodal": "Multimodal",
}
colores = {
    "nir": "#1f77b4",
    "brix": "#ff7f0e",
    "cnn": "#2ca02c",
    "multimodal": "#d62728",
}

# 1. Panel principal: ROC, Precision-Recall, Distribucion y F1 vs Umbral
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Evaluación Out-of-Fold (5-Fold CV) - Granulación Citrus sinensis (n=100)",
    fontsize=13,
    fontweight="bold",
)

# 1a. Curvas ROC
ax_roc = axes[0, 0]
for m in modelos:
  fpr, tpr, _ = roc_curve(y, data[m])
  auc = roc_auc_score(y, data[m])
  ax_roc.plot(
      fpr, tpr, lw=2, color=colores[m], label=f"{etiquetas[m]} (AUC = {auc:.3f})"
  )
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")
ax_roc.set_title("Curvas ROC (OOF)")
ax_roc.set_xlabel("1 - Especificidad (FPR)")
ax_roc.set_ylabel("Recall (TPR)")
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.grid(alpha=0.3)

# 1b. Curvas Precision-Recall
ax_pr = axes[0, 1]
prevalencia = y.mean()
for m in modelos:
  prec, rec, _ = precision_recall_curve(y, data[m])
  ap = average_precision_score(y, data[m])
  ax_pr.plot(
      rec, prec, lw=2, color=colores[m], label=f"{etiquetas[m]} (AP = {ap:.3f})"
  )
ax_pr.axhline(
    prevalencia,
    color="k",
    linestyle="--",
    lw=1,
    label=f"Prevalencia ({prevalencia:.2f})",
)
ax_pr.set_title("Curvas Precision-Recall (OOF)")
ax_pr.set_xlabel("Recall (Clase 1)")
ax_pr.set_ylabel("Precision (Clase 1)")
ax_pr.legend(loc="upper right", fontsize=9)
ax_pr.grid(alpha=0.3)

# 1c. Boxplots + puntos de probabilidad por clase
ax_dist = axes[1, 0]
rng = np.random.default_rng(42)
pos_base = np.arange(len(modelos)) * 2.4

for idx, m in enumerate(modelos):
  p0 = data[m][y == 0]
  p1 = data[m][y == 1]

  bp0 = ax_dist.boxplot(
      [p0], positions=[pos_base[idx] - 0.4], widths=0.65, patch_artist=True
  )
  bp1 = ax_dist.boxplot(
      [p1], positions=[pos_base[idx] + 0.4], widths=0.65, patch_artist=True
  )
  for b in bp0["boxes"]:
    b.set_facecolor("#aec7e8")
  for b in bp1["boxes"]:
    b.set_facecolor("#ff9896")

  ax_dist.scatter(
      pos_base[idx] - 0.4 + rng.normal(0, 0.06, size=len(p0)),
      p0,
      color="#1f77b4",
      alpha=0.35,
      s=14,
  )
  ax_dist.scatter(
      pos_base[idx] + 0.4 + rng.normal(0, 0.06, size=len(p1)),
      p1,
      color="#d62728",
      alpha=0.85,
      s=24,
      marker="D",
  )

ax_dist.axhline(0.5, color="gray", linestyle=":", lw=1.2)
ax_dist.set_xticks(pos_base)
ax_dist.set_xticklabels([etiquetas[m] for m in modelos])
ax_dist.set_ylim(-0.05, 1.05)
ax_dist.set_title("Probabilidades OOF: Sano (0, azul) vs Granulado (1, rojo)")
ax_dist.set_ylabel("Probabilidad predicha")
ax_dist.grid(axis="y", alpha=0.3)

# 1d. F1-score vs umbral
ax_th = axes[1, 1]
umbrales = np.linspace(0.10, 0.80, 71)
for m in modelos:
  f1s = [
      f1_score(y, (data[m] >= th).astype(int), zero_division=0)
      for th in umbrales
  ]
  ax_th.plot(umbrales, f1s, lw=2, color=colores[m], label=etiquetas[m])
ax_th.axvline(0.5, color="gray", linestyle=":", lw=1.2, label="Umbral 0.50")
ax_th.set_title("F1-Score (Clase 1) según Umbral")
ax_th.set_xlabel("Umbral")
ax_th.set_ylabel("F1-Score")
ax_th.legend(loc="upper right", fontsize=9)
ax_th.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
ruta_panel = os.path.join(CARPETA_SALIDA, "panel_oof_comparacion.png")
plt.savefig(ruta_panel, dpi=150)
plt.close(fig)

# 2. Matrices de confusión en umbral 0.50
fig_cm, axes_cm = plt.subplots(1, 4, figsize=(16, 4))
fig_cm.suptitle(
    "Matrices de Confusión OOF (Umbral = 0.50)", fontsize=13, fontweight="bold"
)

for ax, m in zip(axes_cm, modelos):
  preds = (data[m] >= 0.50).astype(int)
  cm = confusion_matrix(y, preds, labels=[0, 1])
  ax.imshow(cm, cmap="Blues")
  ax.set_title(etiquetas[m])
  ax.set_xticks([0, 1])
  ax.set_yticks([0, 1])
  ax.set_xticklabels(["Sano (0)", "Gran. (1)"])
  ax.set_yticklabels(["Sano (0)", "Gran. (1)"])
  ax.set_xlabel("Predicción")
  ax.set_ylabel("Real")
  for i in range(2):
    for j in range(2):
      val = cm[i, j]
      color_txt = "white" if val > cm.max() * 0.5 else "black"
      ax.text(
          j,
          i,
          str(val),
          ha="center",
          va="center",
          color=color_txt,
          fontsize=12,
          fontweight="bold",
      )

plt.tight_layout(rect=[0, 0, 1, 0.92])
ruta_cm = os.path.join(CARPETA_SALIDA, "matrices_confusion_oof.png")
plt.savefig(ruta_cm, dpi=150)
plt.close(fig_cm)

print("Figuras guardadas en:", CARPETA_SALIDA)