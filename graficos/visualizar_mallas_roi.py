import math
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
  sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np

from graficos.utilidades_graficos import guardar_figura

CARPETA_DATOS = RAIZ / "datos"
CARPETA_SALIDA = RAIZ / "graficos" / "inspeccion_roi"


def guardar_mallas(
    X_data: np.ndarray,
    y_data: np.ndarray,
    ids_data: np.ndarray,
    canal: int = 2,
    nombre_canal: str = "N",
    por_pagina: int = 25,
) -> None:
  CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
  total = len(X_data)
  num_paginas = math.ceil(total / por_pagina)
  cols = 5
  rows = math.ceil(por_pagina / cols)

  for p in range(num_paginas):
    inicio = p * por_pagina
    fin = min(inicio + por_pagina, total)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(len(axes)):
      ax = axes[idx]
      muestra_idx = inicio + idx
      if muestra_idx < fin:
        # canal 0 = RGB, 1 = UV, 2 = NIR
        img = X_data[muestra_idx, :, :, canal]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{ids_data[muestra_idx]} (#{muestra_idx + 1}) | Etq: {y_data[muestra_idx]}",
            fontsize=9,
        )
      ax.axis("off")

    plt.tight_layout()
    salida = CARPETA_SALIDA / f"malla_{nombre_canal}_parte_{p + 1}.png"
    guardar_figura(fig, salida)


def main() -> None:
  X = np.load(CARPETA_DATOS / "X_dataset_multicanal.npy")
  y = np.load(CARPETA_DATOS / "y_dataset_etiquetas.npy")
  ids = np.load(CARPETA_DATOS / "ids_imagenes.npy")

  guardar_mallas(X, y, ids, canal=0, nombre_canal="RGB", por_pagina=25)
  guardar_mallas(X, y, ids, canal=1, nombre_canal="UV", por_pagina=25)
  guardar_mallas(X, y, ids, canal=2, nombre_canal="N", por_pagina=25)


if __name__ == "__main__":
  main()
