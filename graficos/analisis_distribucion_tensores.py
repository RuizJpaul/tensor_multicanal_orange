from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
  sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np

from graficos.utilidades_graficos import guardar_figura

CARPETA_DATOS = RAIZ / "datos"
CARPETA_SALIDA = RAIZ / "graficos" / "graficos_tensor"

NOMBRES_CANALES = ["RGB (Canal 0)", "UV (Canal 1)", "NIR (Canal 2)"]
COLORES_CANALES = ["#d95f02", "#7570b3", "#1b9e77"]


def cargar_datos():
  X = np.load(CARPETA_DATOS / "X_dataset_multicanal.npy")
  y = np.load(CARPETA_DATOS / "y_dataset_etiquetas.npy")
  ids_path = CARPETA_DATOS / "ids_imagenes.npy"
  if ids_path.exists():
    ids = np.load(ids_path)
  else:
    ids = np.array([f"M{i + 1}" for i in range(len(X))])
  return X, y, ids


def graficar_distribucion_clases(y: np.ndarray) -> None:
  clases, conteos = np.unique(y, return_counts=True)
  etiquetas = [f"Clase {c}\n({'Sano' if c == 0 else 'Granulado'})" for c in clases]
  porcentajes = (conteos / len(y)) * 100

  fig, ax = plt.subplots(figsize=(6.5, 4.5))
  barras = ax.bar(
      etiquetas, conteos, color=["#4c78a8", "#e45756"], width=0.5, edgecolor="black"
  )
  ax.set_title(f"Distribución de Etiquetas del Dataset (n = {len(y)})", fontweight="bold")
  ax.set_ylabel("Número de muestras")
  ax.set_ylim(0, max(conteos) * 1.18)
  ax.grid(axis="y", linestyle="--", alpha=0.35)

  for barra, cnt, pct in zip(barras, conteos, porcentajes):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + max(conteos) * 0.025,
        f"{cnt} ({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

  plt.tight_layout()
  guardar_figura(fig, CARPETA_SALIDA / "01_distribucion_clases.png")


def graficar_histogramas_canales(X: np.ndarray) -> None:
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
  fig.suptitle("Distribución de Intensidades Normalizadas por Canal", fontweight="bold")

  for c in range(X.shape[-1]):
    vals = X[:, :, :, c].ravel()
    ax = axes[c]
    ax.hist(vals, bins=50, range=(0.0, 1.0), color=COLORES_CANALES[c], alpha=0.85)
    ax.axvline(vals.mean(), color="black", linestyle="--", lw=1.4, label=f"Media: {vals.mean():.3f}")
    ax.axvline(np.median(vals), color="#222222", linestyle=":", lw=1.4, label=f"Mediana: {np.median(vals):.3f}")
    ax.set_title(NOMBRES_CANALES[c])
    ax.set_xlabel("Intensidad normalizada [0, 1]")
    if c == 0:
      ax.set_ylabel("Frecuencia de píxeles")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, CARPETA_SALIDA / "02_histogramas_canales.png")


def graficar_perfil_por_muestra(X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> None:
  # Media y desviación estándar por muestra y canal: shape (N, 3)
  medias = X.mean(axis=(1, 2))
  stds = X.std(axis=(1, 2))
  indices = np.arange(len(X))

  fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
  fig.suptitle("Media y Dispersión de Intensidad por Muestra (Detección de Atípicos)", fontweight="bold")

  for c in range(X.shape[-1]):
    ax = axes[c]
    m_c = medias[:, c]
    s_c = stds[:, c]

    ax.plot(indices, m_c, color=COLORES_CANALES[c], lw=1.6, marker="o", markersize=3.5, label="Media ROI")
    ax.fill_between(indices, np.clip(m_c - s_c, 0, 1), np.clip(m_c + s_c, 0, 1),
                    color=COLORES_CANALES[c], alpha=0.15, label="±1 Desv. Est.")

    # Resaltar muestras con etiqueta 1
    idx_gran = np.where(y == 1)[0]
    ax.scatter(idx_gran, m_c[idx_gran], color="#d62728", s=26, zorder=4, label="Clase 1 (Granulado)")

    # Marcar valores atípicos fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    q1, q3 = np.percentile(m_c, [25, 75])
    iqr = q3 - q1
    bajo = q1 - 1.5 * iqr
    alto = q3 + 1.5 * iqr
    atipicos = np.where((m_c < bajo) | (m_c > alto))[0]
    if len(atipicos) > 0:
      ax.scatter(atipicos, m_c[atipicos], facecolors="none", edgecolors="black", s=70, lw=1.5, zorder=5)
      for idx_a in atipicos:
        ax.annotate(
            f"{ids[idx_a]} ({m_c[idx_a]:.3f})",
            xy=(idx_a, m_c[idx_a]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel(NOMBRES_CANALES[c])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if c == 0:
      ax.legend(loc="upper right", fontsize=8, ncol=3)

  paso = max(1, len(X) // 20)
  axes[-1].set_xticks(indices[::paso])
  axes[-1].set_xticklabels(ids[::paso], rotation=45, ha="right", fontsize=8.5)
  axes[-1].set_xlabel("Muestra")

  plt.tight_layout(rect=[0, 0, 1, 0.95])
  guardar_figura(fig, CARPETA_SALIDA / "03_perfil_muestras_y_atipicos.png")


def graficar_comparacion_por_clase(X: np.ndarray, y: np.ndarray) -> None:
  medias = X.mean(axis=(1, 2))  # (N, 3)

  fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
  fig.suptitle("Intensidad Media por Muestra según Clase (0 vs 1)", fontweight="bold")

  for c in range(X.shape[-1]):
    ax = axes[c]
    datos_0 = medias[y == 0, c]
    datos_1 = medias[y == 1, c]

    bp = ax.boxplot(
        [datos_0, datos_1],
        tick_labels=[f"Sano (0)\nn={len(datos_0)}", f"Granulado (1)\nn={len(datos_1)}"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#a6bddb")
    bp["boxes"][1].set_facecolor("#fdbb84")

    rng = np.random.default_rng(42 + c)
    ax.scatter(1 + rng.normal(0, 0.05, len(datos_0)), datos_0, color="#1f77b4", alpha=0.45, s=18)
    ax.scatter(2 + rng.normal(0, 0.05, len(datos_1)), datos_1, color="#d62728", alpha=0.8, s=24)

    ax.set_title(NOMBRES_CANALES[c])
    ax.set_ylabel("Intensidad media del ROI")
    ax.grid(axis="y", alpha=0.3)

  plt.tight_layout(rect=[0, 0, 1, 0.93])
  guardar_figura(fig, CARPETA_SALIDA / "04_comparacion_clases_por_canal.png")


def graficar_correlacion_canales(X: np.ndarray) -> None:
  # Promedio por muestra para cada canal -> matriz (N, 3)
  medias = X.mean(axis=(1, 2))
  corr = np.corrcoef(medias, rowvar=False)

  fig, ax = plt.subplots(figsize=(5.5, 4.8))
  im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
  cortos = ["RGB", "UV", "NIR"]
  ax.set_xticks(range(3))
  ax.set_yticks(range(3))
  ax.set_xticklabels(cortos)
  ax.set_yticklabels(cortos)
  ax.set_title("Correlación entre Canales (Media por ROI)", fontweight="bold")

  for i in range(3):
    for j in range(3):
      val = corr[i, j]
      txt_color = "white" if abs(val) > 0.6 else "black"
      ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlación de Pearson")
  plt.tight_layout()
  guardar_figura(fig, CARPETA_SALIDA / "05_correlacion_canales.png")


def main() -> None:
  X, y, ids = cargar_datos()
  print(f"Cargado tensor X: {X.shape} | y: {y.shape} | ids: {len(ids)}")

  graficar_distribucion_clases(y)
  graficar_histogramas_canales(X)
  graficar_perfil_por_muestra(X, y, ids)
  graficar_comparacion_por_clase(X, y)
  graficar_correlacion_canales(X)
  print(f"Listo. Revisa la carpeta: {CARPETA_SALIDA}")


if __name__ == "__main__":
  main()