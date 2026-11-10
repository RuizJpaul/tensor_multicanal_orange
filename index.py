import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar directamente el archivo Excel con las evaluaciones visuales
excel_path = 'resultados_granulacion.xlsx'
df_etiquetas = pd.read_excel(excel_path)

def procesar_tensor_multicanal(path_rgb, path_uv, path_nir, target_size=(224, 224)):
    """
    Procesa, segmenta (ROI), redimensiona y apila las imágenes RGB, UV y NIR 
    para formar el tensor tridimensional (H x W x 3) para la CNN.
    """
    # Cargar las imágenes en escala original
    img_rgb = cv2.imread(path_rgb)
    img_uv  = cv2.imread(path_uv)
    img_nir = cv2.imread(path_nir)
    
    if img_rgb is None or img_uv is None or img_nir is None:
        raise ValueError("Error al cargar una o más rutas de imágenes. Verifica los archivos.")

    # Preprocesamiento y Extracción de la Región de Interés (ROI) adaptativa
    def extraer_roi(img, es_fondo_oscuro=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if es_fondo_oscuro:
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return img[y:y+h, x:x+w]
        return img 

    roi_rgb = extraer_roi(img_rgb, es_fondo_oscuro=False)
    roi_uv  = extraer_roi(img_uv,  es_fondo_oscuro=True)
    roi_nir = extraer_roi(img_nir, es_fondo_oscuro=True)

    # Redimensionamiento y normalización
    roi_rgb_resized = cv2.resize(roi_rgb, target_size)
    roi_uv_resized  = cv2.resize(roi_uv, target_size)
    roi_nir_resized = cv2.resize(roi_nir, target_size)

    channel_rgb = cv2.cvtColor(roi_rgb_resized, cv2.COLOR_BGR2GRAY)
    channel_uv  = cv2.cvtColor(roi_uv_resized, cv2.COLOR_BGR2GRAY)
    channel_nir = cv2.cvtColor(roi_nir_resized, cv2.COLOR_BGR2GRAY)

    channel_rgb_norm = channel_rgb.astype(np.float32) / 255.0
    channel_uv_norm  = channel_uv.astype(np.float32) / 255.0
    channel_nir_norm = channel_nir.astype(np.float32) / 255.0

    tensor_tridimensional = np.stack([channel_rgb_norm, channel_uv_norm, channel_nir_norm], axis=-1)
    return tensor_tridimensional

def construir_dataset_completo(directorio_muestras, df_labels, total_muestras=50):
    X_data = []
    y_data = []
    
    print("Iniciando la construcción del dataset multicanal y vinculación de etiquetas...")
    
    for i in range(1, total_muestras + 1):
        sample_id = f"M{i}"
        path_rgb = os.path.join(directorio_muestras, f"{sample_id}_RGB.jpg")
        path_uv  = os.path.join(directorio_muestras, f"{sample_id}_UV.jpg")
        path_nir = os.path.join(directorio_muestras, f"{sample_id}_N.jpg")
        
        # Validar que existan los 3 archivos de la muestra actual
        if os.path.exists(path_rgb) and os.path.exists(path_uv) and os.path.exists(path_nir):
            # Buscar la etiqueta correspondiente en el DataFrame del Excel
            fila = df_labels[df_labels['ID'] == sample_id]
            
            if not fila.empty:
                etiqueta_real = int(fila['Granulación'].values[0])
                try:
                    tensor_multicanal = procesar_tensor_multicanal(path_rgb, path_uv, path_nir)
                    X_data.append(tensor_multicanal)
                    y_data.append(etiqueta_real)
                except Exception as e:
                    print(f"Error procesando la muestra {sample_id}: {e}")
            else:
                print(f"Advertencia: No se encontró etiqueta para {sample_id} en el Excel.")
        else:
            print(f"Advertencia: Faltan archivos de imagen para la muestra {sample_id}, omitiendo...")
            
    # Convertir a arrays de NumPy optimizados para Deep Learning
    X_dataset = np.array(X_data, dtype=np.float32)
    y_dataset = np.array(y_data, dtype=np.int32)
    
    return X_dataset, y_dataset

# --- EJECUCIÓN DEL LOTE COMPLETO ---
X_train, y_train = construir_dataset_completo('muestras/', df_etiquetas, total_muestras=50)

print("\n¡Dataset multicanal y etiquetas listos!")
print(f"Forma de la matriz X (Tensores): {X_train.shape}")       # Esperado: (50, 224, 224, 3)
print(f"Forma del vector y (Etiquetas): {y_train.shape}")      # Esperado: (50,)
print(f"Distribución de clases (Sanos 0 vs Dañados 1): {np.bincount(y_train)}")

# Guardar los archivos definitivos en disco
np.save('X_dataset_multicanal.npy', X_train)
np.save('y_dataset_etiquetas.npy', y_train)
print("Archivos 'X_dataset_multicanal.npy' y 'y_dataset_etiquetas.npy' guardados correctamente.")