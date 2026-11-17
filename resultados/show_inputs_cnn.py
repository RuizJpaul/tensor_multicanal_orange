import os
import matplotlib.pyplot as plt
import numpy as np

print("Cargando datasets para análisis...")
X = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

# ==========================================
# 1. ESTADÍSTICAS Y DISTRIBUCIÓN DE CLASES
# ==========================================
print("\n--- RESUMEN ESTADÍSTICO DEL DATASET ---")
print(f"Dimensiones del Tensor X (Imágenes Multicanal): {X.shape}")
print(f"Dimensiones del Vector y (Etiquetas): {y.shape}")
print(f"Rango de valores de píxeles: Mín = {X.min():.2f}, Máx = {X.max():.2f}")

clases, conteos = np.unique(y, return_counts=True)
print("\nDistribución de Clases:")
for c, count in zip(clases, conteos.astype(int)):
    nombre_clase = "Sano (0)" if c == 0 else "Granulado / Dañado (1)"
    porcentaje = (count / len(y)) * 100
    print(f" - Clase {c} [{nombre_clase}]: {count} muestras ({porcentaje:.1f}%)")

# ==========================================
# 2. GENERAR GRÁFICO COMPARATIVO DE MUESTRAS
# ==========================================
os.makedirs('graficos_tesis', exist_ok=True)

# Encontrar índices de ejemplo para cada clase
idx_sano = np.where(y == 0)[0][0] if 0 in y else 0
idx_granulado = np.where(y == 1)[0][0] if 1 in y else 0

plt.figure(figsize=(8, 4))

# Muestra Sana
plt.subplot(1, 2, 1)
img_sano = X[idx_sano]
if img_sano.max() <= 1.0:
    img_sano = np.clip(img_sano, 0, 1)
plt.imshow(img_sano)
plt.title(f'Muestra Sana (Clase 0)\nÍndice #{idx_sano}', fontsize=10, fontweight='bold')
plt.axis('off')

# Muestra Granulada
plt.subplot(1, 2, 2)
img_gran = X[idx_granulado]
if img_gran.max() <= 1.0:
    img_gran = np.clip(img_gran, 0, 1)
plt.imshow(img_gran)
plt.title(f'Muestra Granulada (Clase 1)\nÍndice #{idx_granulado}', fontsize=10, fontweight='bold')
plt.axis('off')

plt.suptitle('Comparativa Visual del Dataset Tridimensional (ROI)', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graficos_tesis/fig6_comparativa_clases_dataset.png', dpi=300)
plt.close()

print("\n¡Gráfico comparativo guardado exitosamente en 'graficos_tesis/fig6_comparativa_clases_dataset.png'!")