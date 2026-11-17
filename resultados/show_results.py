import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración visual profesional para publicaciones científicas / Tesis
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

print("1. Cargando datos y modelo multimodal guardado...")
df_nir = pd.read_excel('Dataset.xlsx', sheet_name='Dataset_Nir')
df_brix = pd.read_excel('Dataset.xlsx', sheet_name='Brix')

cols_nir = [c for c in df_nir.columns if c not in ['ID', 'Etiqueta (granulación)']]
X_spectral = df_nir[cols_nir].values
X_brix = df_brix[['Promedio']].values
X_images = np.load('X_dataset_multicanal.npy')
y = np.load('y_dataset_etiquetas.npy')

# Normalización
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

# Predicciones
y_pred_prob = modelo.predict([X_spec_test, X_brix_test, X_img_test])
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Crear carpeta para guardar los gráficos
os.makedirs('graficos_tesis', exist_ok=True)
print("\nGenerando gráficos en la carpeta 'graficos_tesis'...")

# ==========================================
# FIGURA 1: MATRIZ DE CONFUSIÓN
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Sano (0)', 'Granulado (1)'],
            yticklabels=['Sano (0)', 'Granulado (1)'],
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Matriz de Confusión - Modelo Multimodal', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Clase Predicha por el Modelo', labelpad=10)
plt.ylabel('Clase Real (Ground Truth)', labelpad=10)
plt.tight_layout()
plt.savefig('graficos_tesis/fig1_matriz_confusion.png', dpi=300)
plt.close()
print(" -> FIGURA 1: Matriz de Confusión guardada.")

# ==========================================
# FIGURA 2: BARRAS DE MÉTRICAS GLOBALES
# ==========================================
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metricas_valores = [
    report['accuracy'],
    report['weighted avg']['precision'],
    report['weighted avg']['recall'],
    report['weighted avg']['f1-score']
]

plt.figure(figsize=(7, 4.5))
colores = ['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1']
bars = plt.bar(metricas_nombres, metricas_valores, color=colores, width=0.5, edgecolor='black', linewidth=0.8)

plt.ylim(0, 1.15)
plt.ylabel('Puntaje (0.0 - 1.0)', labelpad=10)
plt.title('Métricas Globales de Evaluación del Clasificador Multimodal', fontsize=12, fontweight='bold', pad=12)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.03,
             f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos_tesis/fig2_metricas_globales.png', dpi=300)
plt.close()
print(" -> FIGURA 2: Métricas globales guardadas.")

# ==========================================
# FIGURA 3: DIAGRAMA DEL TENSOR TRIDIMENSIONAL
# ==========================================
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection='3d')

# Dimensiones representativas (Alto, Ancho, Canales)
H, W, C = 224, 224, 3
x = np.linspace(0, 1, 10)
y_grid = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y_grid)

# Dibujar capas de canales (RGB/UV/NIR)
colors = ['#d95f02', '#7570b3', '#1b9e77']
channel_labels = ['Canal 1 (RGB / Pulpa)', 'Canal 2 (UV / Contraste)', 'Canal 3 (NIR / Densidad)']

for i in range(C):
    Z = np.full_like(X, i * 1.5)
    ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.6, edgecolor='k', linewidth=0.3)
    ax.text(0.5, 0.5, i * 1.5 + 0.2, channel_labels[i], color='black', fontsize=9, fontweight='bold', ha='center')

ax.set_title('Representación del Tensor Tridimensional ROI ($H \\times W \\times C$)', fontsize=11, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('graficos_tesis/fig3_tensor_tridimensional.png', dpi=300)
plt.close()
print(" -> FIGURA 3: Diagrama conceptual del Tensor Tridimensional guardado.")

# ==========================================
# FIGURA 4: ESQUEMA DEL DIAGRAMA DE CAPAS DE LA CNN
# ==========================================
capas_cnn = [
    'Input Image\n(224x224x3)',
    'Conv2D (32)\n+ ReLU',
    'MaxPool2D\n(112x112x32)',
    'Conv2D (64)\n+ ReLU',
    'MaxPool2D\n(56x56x64)',
    'Conv2D (128)\n+ ReLU',
    'GlobalAvgPool\n(128)',
    'Dense Feat\n(32)'
]

plt.figure(figsize=(10, 3.5))
for i, capa in enumerate(capas_cnn):
    plt.text(i * 1.5, 0.5, capa, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#e6f2ff', edgecolor='#1f77b4', lw=1.5),
             fontsize=8.5, fontweight='bold')
    if i < len(capas_cnn) - 1:
        plt.annotate('', xy=((i + 1) * 1.5 - 0.45, 0.5), xytext=(i * 1.5 + 0.45, 0.5),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color='#333333'))

plt.xlim(-0.8, (len(capas_cnn) - 1) * 1.5 + 0.8)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Flujo Metodológico de la Rama Red Convolucional (CNN)', fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('graficos_tesis/fig4_diagrama_flujo_cnn.png', dpi=300)
plt.close()
print(" -> FIGURA 4: Diagrama de flujo de la CNN guardado.")

print("\n¡Proceso finalizado! Todos los gráficos fueron generados en alta resolución (300 DPI) dentro de 'graficos_tesis/'.")