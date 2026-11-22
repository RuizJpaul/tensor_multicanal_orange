import math
import os
import matplotlib.pyplot as plt
import numpy as np

X = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

os.makedirs('inspeccion_roi', exist_ok=True)


def guardar_mallas(X_data, y_data, canal=0, nombre_canal='N', por_pagina=25):
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
        # En tu tensor: canal 0 = RGB, 1 = UV, 2 = NIR (en escala de grises normalizada)
        img = X_data[muestra_idx, :, :, canal]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(
            f'#{muestra_idx + 1} | Etq: {y_data[muestra_idx]}', fontsize=9
        )
      ax.axis('off')

    plt.tight_layout()
    salida = f'inspeccion_roi/malla_{nombre_canal}_parte_{p + 1}.png'
    plt.savefig(salida, dpi=150)
    plt.close(fig)
    print(f'Guardada: {salida}')


guardar_mallas(X, y, canal=2, nombre_canal='N', por_pagina=25)