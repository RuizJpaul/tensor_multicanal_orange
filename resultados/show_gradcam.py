import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("1. Cargando datos y modelo multimodal...")
X_images = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

if X_images.max() > 1.0:
    X_images = X_images.astype('float32') / 255.0

_, idx_test, _, _ = train_test_split(
    np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
)
X_img_test = X_images[idx_test]

# Cargar el modelo guardado
modelo = tf.keras.models.load_model('modelo_multimodal_valencia.h5')

# ==========================================
# 2. CONFIGURACIÓN DE GRAD-CAM
# ==========================================
# Última capa convolucional de la rama visual de tu modelo
last_conv_layer_name = 'cnn_conv3' 
last_conv_layer = modelo.get_layer(last_conv_layer_name)

# Crear un submodelo que mapee las entradas a la salida de la última conv y la salida final
grad_model = tf.keras.models.Model(
    inputs=modelo.inputs,
    outputs=[last_conv_layer.output, modelo.output]
)

# Seleccionar una muestra de prueba del conjunto de test
sample_idx = 0
img_sample = X_img_test[sample_idx:sample_idx+1] # (1, 224, 224, 3)
spec_sample = np.zeros((1, 24))  # Dummy input para la rama espectral
brix_sample = np.zeros((1, 1))   # Dummy input para la rama brix

print("Calculando gradientes y mapas de calor (Grad-CAM)...")
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model([spec_sample, brix_sample, img_sample])
    loss = predictions[:, 0]

# Extraer gradientes de la última capa convolucional
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
pooled_grads = pooled_grads.numpy()
conv_outputs = conv_outputs.numpy()

# Ponderar los canales del mapa de características
for i in range(pooled_grads.shape[-1]):
    conv_outputs[:, :, i] *= pooled_grads[i]

# Generar el mapa de calor final
heatmap = np.mean(conv_outputs, axis=-1)
heatmap = np.maximum(heatmap, 0)
if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

# ==========================================
# 3. GRAFICAR Y GUARDAR RESULTADO
# ==========================================
os.makedirs('graficos_tesis', exist_ok=True)
plt.figure(figsize=(9, 4.5))

# Imagen Original
plt.subplot(1, 2, 1)
img_show = img_sample[0]
plt.imshow(img_show)
plt.title('Región de Interés (ROI) Original', fontsize=11, fontweight='bold')
plt.axis('off')

# Superposición Grad-CAM
plt.subplot(1, 2, 2)
plt.imshow(img_show)
plt.imshow(heatmap, cmap='jet', alpha=0.5) # Mapa de calor translúcido
plt.title('Mapa de Activación Grad-CAM\n(Zona de Enfoque de la CNN)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.suptitle('Explicabilidad Espacial de la Red Convolucional', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graficos_tesis/fig7_gradcam_cnn.png', dpi=300)
plt.close()

print("¡Gráfico Grad-CAM generado exitosamente en 'graficos_tesis/fig7_gradcam_cnn.png'!")