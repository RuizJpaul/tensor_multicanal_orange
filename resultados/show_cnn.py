import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("1. Cargando datos de imágenes y modelo...")
X_images = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

if X_images.max() > 1.0:
    X_images = X_images.astype('float32') / 255.0

_, idx_test, _, _ = train_test_split(
    np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
)
X_img_test = X_images[idx_test]

# Cargar el modelo multimodal completo
modelo = tf.keras.models.load_model('modelo_multimodal_valencia.h5')

# ==========================================
# 2. CREAR UN SUB-MODELO PARA EXTRAER LA CAPA CONVOLUCIONAL
# ==========================================
capa_objetivo = 'cnn_conv1'

# Usamos modelo.inputs para capturar todas las entradas de la red multimodal
extractor_caracteristicas = tf.keras.Model(
    inputs=modelo.inputs,
    outputs=modelo.get_layer(capa_objetivo).output
)

# Seleccionamos una imagen de prueba
imagen_muestra = X_img_test[0:1] # Forma: (1, 224, 224, 3)

# Obtenemos los mapas de características (pasando también entradas vacías para espectral y brix)
features = extractor_caracteristicas.predict([
    np.zeros((1, 24)),  # Dummy spectral input
    np.zeros((1, 1)),   # Dummy brix input
    imagen_muestra
])

# ==========================================
# 3. GRAFICAR Y GUARDAR LOS MAPAS DE ACTIVACIÓN
# ==========================================
os.makedirs('graficos_tesis', exist_ok=True)
n_filtros = 16  # Primeros 16 filtros
plt.figure(figsize=(12, 12))

for i in range(n_filtros):
    plt.subplot(4, 4, i + 1)
    plt.imshow(features[0, :, :, i], cmap='viridis')
    plt.axis('off')
    plt.title(f'Filtro {i+1}', fontsize=9)

plt.suptitle(f'Mapas de Características extraídos en [{capa_objetivo}]', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos_tesis/fig5_mapas_caracteristicas_cnn.png', dpi=300)
plt.close()

print("¡Gráfico generado con éxito! Revisa la carpeta 'graficos_tesis/fig5_mapas_caracteristicas_cnn.png'.")