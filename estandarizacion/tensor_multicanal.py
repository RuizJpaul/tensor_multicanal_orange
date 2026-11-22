import os
import cv2
import numpy as np
import pandas as pd


def procesar_tensor_multicanal(path_rgb, path_uv, path_nir, target_size=(224, 224)):
  img_rgb = cv2.imread(path_rgb)
  img_uv = cv2.imread(path_uv)
  img_nir = cv2.imread(path_nir)

  if img_rgb is None or img_uv is None or img_nir is None:
    raise ValueError("Error al cargar una o más rutas de imágenes.")

  def extraer_roi(img, es_fondo_oscuro=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if es_fondo_oscuro:
      _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    else:
      _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
      c = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(c)
      return img[y : y + h, x : x + w]
    return img

  roi_rgb = extraer_roi(img_rgb, es_fondo_oscuro=False)
  roi_uv = extraer_roi(img_uv, es_fondo_oscuro=True)
  roi_nir = extraer_roi(img_nir, es_fondo_oscuro=True)

  roi_rgb_resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_AREA)
  roi_uv_resized = cv2.resize(roi_uv, target_size, interpolation=cv2.INTER_AREA)
  roi_nir_resized = cv2.resize(roi_nir, target_size, interpolation=cv2.INTER_AREA)

  channel_rgb = cv2.cvtColor(roi_rgb_resized, cv2.COLOR_BGR2GRAY).astype(
      np.float32
  ) / 255.0
  channel_uv = cv2.cvtColor(roi_uv_resized, cv2.COLOR_BGR2GRAY).astype(
      np.float32
  ) / 255.0
  channel_nir = cv2.cvtColor(roi_nir_resized, cv2.COLOR_BGR2GRAY).astype(
      np.float32
  ) / 255.0

  return np.stack([channel_rgb, channel_uv, channel_nir], axis=-1)


def construir_dataset_completo(
    directorio_muestras="muestras_filtros",
    excel_path="resultados_granulacion.xlsx",
    total_muestras=100,
):
  df_etiquetas = pd.read_excel(excel_path)
  df_etiquetas["ID"] = df_etiquetas["ID"].astype(str).str.strip()
  df_etiquetas = df_etiquetas.set_index("ID")

  X_data, y_data, ids_validos = [], [], []

  for i in range(1, total_muestras + 1):
    sample_id = f"M{i}"
    path_rgb = os.path.join(directorio_muestras, f"{sample_id}_RGB.jpg")
    path_uv = os.path.join(directorio_muestras, f"{sample_id}_UV.jpg")
    path_nir = os.path.join(directorio_muestras, f"{sample_id}_N.jpg")

    if (
        os.path.exists(path_rgb)
        and os.path.exists(path_uv)
        and os.path.exists(path_nir)
    ):
      if sample_id in df_etiquetas.index:
        etiqueta_real = int(df_etiquetas.loc[sample_id, "Granulación"])
        try:
          tensor = procesar_tensor_multicanal(path_rgb, path_uv, path_nir)
          X_data.append(tensor)
          y_data.append(etiqueta_real)
          ids_validos.append(sample_id)
        except Exception as e:
          print(f"[Error] {sample_id}: {e}")
      else:
        print(f"[Aviso] Sin etiqueta para {sample_id}")
    else:
      print(f"[Aviso] Faltan imágenes para {sample_id}")

  X_dataset = np.array(X_data, dtype=np.float32)
  y_dataset = np.array(y_data, dtype=np.int32)
  ids_array = np.array(ids_validos)

  np.save("X_dataset_multicanal.npy", X_dataset)
  np.save("y_dataset_etiquetas.npy", y_dataset)
  np.save("ids_imagenes.npy", ids_array)

  print(f"Tensores: {X_dataset.shape} | Etiquetas: {y_dataset.shape}")
  print(f"Distribución de clases (0 vs 1): {np.bincount(y_dataset)}")


if __name__ == "__main__":
  construir_dataset_completo()