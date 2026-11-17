import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("1. Cargando datos y modelo multimodal...")
df_nir = pd.read_excel('Dataset.xlsx', sheet_name='Dataset_Nir')
df_brix = pd.read_excel('Dataset.xlsx', sheet_name='Brix')

cols_nir = [c for c in df_nir.columns if c not in ['ID', 'Etiqueta (granulación)']]
X_spectral = df_nir[cols_nir].values
X_brix = df_brix[['Promedio']].values
X_images = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

# Preprocesamiento
scaler_spec = StandardScaler()
X_spec_scaled = scaler_spec.fit_transform(X_spectral)
scaler_brix = StandardScaler()
X_brix_scaled = scaler_brix.fit_transform(X_brix)
if X_images.max() > 1.0:
    X_images = X_images.astype('float32') / 255.0

indices = np.arange(len(y))
_, idx_test, _, y_test = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

X_spec_test = X_spec_scaled[idx_test]
X_brix_test = X_brix_scaled[idx_test]
X_img_test = X_images[idx_test]

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('modelo_multimodal_valencia.h5')

# ==========================================
# 2. CREAR EXTRACTORES PARA CADA RAMA Y LA FUSIÓN
# ==========================================
# Extraemos las salidas de las capas densas previas a la fusión y la fusión misma
extractor_features = tf.keras.Model(
    inputs=modelo.inputs,
    outputs=[
        modelo.get_layer('nir_dense_2').output,     # Rama Espectral (32 dims)
        modelo.get_layer('brix_dense_2').output,    # Rama Brix (8 dims)
        modelo.get_layer('cnn_dense_feat').output,  # Rama Visual/CNN (32 dims)
        modelo.get_layer('capa_fusion').output      # Capa de Fusión (72 dims)
    ]
)

print("Extrayendo representaciones internas de prueba...")
feat_nir, feat_brix, feat_cnn, feat_fusion = extractor_features.predict([
    X_spec_test, X_brix_test, X_img_test
])

# ==========================================
# 3. REDUCCIÓN DE DIMENSIONALIDAD (t-SNE)
# ==========================================
print("Aplicando t-SNE para proyectar a 2D...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(y_test)-1))

tsne_nir = tsne.fit_transform(feat_nir)
tsne_brix = tsne.fit_transform(feat_brix)
tsne_cnn = tsne.fit_transform(feat_cnn)
tsne_fusion = tsne.fit_transform(feat_fusion)

# ==========================================
# 4. GRAFICAR COMPARATIVA (2x2 Grid)
# ==========================================
os.makedirs('graficos_tesis', exist_ok=True)
plt.figure(figsize=(12, 10))

nombres_ramas = [
    ('Rama Espectral (NIR)', tsne_nir),
    ('Rama Físico-Química (Brix)', tsne_brix),
    ('Rama Visual (CNN)', tsne_cnn),
    ('Capa de Fusión Multimodal', tsne_fusion)
]

for i, (nombre, datos_2d) in enumerate(nombres_ramas):
    plt.subplot(2, 2, i + 1)
    scatter = plt.scatter(datos_2d[:, 0], datos_2d[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=70)
    plt.title(nombre, fontsize=11, fontweight='bold')
    plt.xlabel('Dimensión t-SNE 1')
    plt.ylabel('Dimensión t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Espacio Latente de Características por Rama vs. Fusión (t-SNE)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('graficos_tesis/fig8_espacio_latente_fusion.png', dpi=300)
plt.close()

print("\n¡Gráfico generado con éxito! Revisa 'graficos_tesis/fig8_espacio_latente_fusion.png'.")