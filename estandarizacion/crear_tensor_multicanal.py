from pathlib import Path
import cv2
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_FOTOS = RAIZ / "datos" / "fotos_rgb_nir_uv"
EXCEL_ETIQUETAS = RAIZ / "datos" / "Etiquetas_granulacion_M1_M100.xlsx"
CARPETA_SALIDA = RAIZ / "datos"
COLUMNAS_ETIQUETA = [
    "Granulación",
    "Granulacion",
    "Etiqueta (granulación)",
    "Etiqueta",
]


def extraer_roi(img: np.ndarray, es_fondo_oscuro: bool = True) -> np.ndarray:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if es_fondo_oscuro:
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
  else:
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

  contornos, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contornos:
    return img

  contorno_mayor = max(contornos, key=cv2.contourArea)
  if cv2.contourArea(contorno_mayor) == 0:
    return img

  x, y, w, h = cv2.boundingRect(contorno_mayor)
  return img[y : y + h, x : x + w]


def procesar_tensor_multicanal(
    path_rgb: Path,
    path_uv: Path,
    path_nir: Path,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
  img_rgb = cv2.imread(str(path_rgb))
  img_uv = cv2.imread(str(path_uv))
  img_nir = cv2.imread(str(path_nir))

  if img_rgb is None or img_uv is None or img_nir is None:
    raise ValueError("Error al cargar una o más imágenes.")

  rois = [
      extraer_roi(img_rgb, es_fondo_oscuro=False),
      extraer_roi(img_uv, es_fondo_oscuro=True),
      extraer_roi(img_nir, es_fondo_oscuro=True),
  ]

  canales = []
  for roi in rois:
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    canales.append(gray)

  return np.stack(canales, axis=-1)


def construir_dataset_completo(total_muestras: int | None = None):
  df_etiquetas = pd.read_excel(EXCEL_ETIQUETAS)
  df_etiquetas.columns = df_etiquetas.columns.astype(str).str.strip()
  df_etiquetas["ID"] = df_etiquetas["ID"].astype(str).str.strip().str.upper()
  df_etiquetas = df_etiquetas.drop_duplicates(subset=["ID"]).set_index("ID")

  col_etiqueta = next((c for c in COLUMNAS_ETIQUETA if c in df_etiquetas.columns), None)
  if col_etiqueta is None:
    raise KeyError(
        f"No se encontró columna de etiqueta en {EXCEL_ETIQUETAS.name}: {list(df_etiquetas.columns)}"
    )

  if total_muestras is None:
    total_muestras = len(df_etiquetas)

  X_data, y_data, ids_validos = [], [], []

  for i in range(1, total_muestras + 1):
    sample_id = f"M{i}"
    path_rgb = CARPETA_FOTOS / f"{sample_id}_RGB.jpg"
    path_uv = CARPETA_FOTOS / f"{sample_id}_UV.jpg"
    path_nir = CARPETA_FOTOS / f"{sample_id}_N.jpg"

    if not (path_rgb.exists() and path_uv.exists() and path_nir.exists()):
      print(f"[Aviso] Faltan imágenes para {sample_id}")
      continue

    if sample_id not in df_etiquetas.index:
      print(f"[Aviso] Sin etiqueta para {sample_id}")
      continue

    etiqueta_num = pd.to_numeric(
        df_etiquetas.loc[sample_id, col_etiqueta], errors="coerce"
    )
    if pd.isna(etiqueta_num):
      print(f"[Error] {sample_id}: etiqueta inválida o vacía")
      continue

    try:
      tensor = procesar_tensor_multicanal(path_rgb, path_uv, path_nir)
      X_data.append(tensor)
      y_data.append(int(etiqueta_num))
      ids_validos.append(sample_id)
    except Exception as e:
      print(f"[Error] {sample_id}: {e}")

  if not X_data:
    print("No se construyó ningún tensor válido.")
    return

  X_dataset = np.array(X_data, dtype=np.float32)
  y_dataset = np.array(y_data, dtype=np.int32)
  ids_array = np.array(ids_validos)

  CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
  np.save(CARPETA_SALIDA / "X_dataset_multicanal.npy", X_dataset)
  np.save(CARPETA_SALIDA / "y_dataset_etiquetas.npy", y_dataset)
  np.save(CARPETA_SALIDA / "ids_imagenes.npy", ids_array)

  print(f"Guardado en: {CARPETA_SALIDA}")
  print(f"Tensores: {X_dataset.shape} | Etiquetas: {y_dataset.shape}")
  print(
      "Distribución de clases:"
      f" {pd.Series(y_dataset).value_counts().sort_index().to_dict()}"
  )


if __name__ == "__main__":
  construir_dataset_completo()