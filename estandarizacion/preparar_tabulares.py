import os
import numpy as np
import pandas as pd

PUNTOS = ["P1", "P2", "P3", "P4"]
CANALES = ["R", "S", "T", "U", "V", "W"]


def aplicar_snv(X):
  """Standard Normal Variate (SNV) fila por fila para firmas espectrales."""
  media = np.mean(X, axis=1, keepdims=True)
  std = np.std(X, axis=1, keepdims=True)
  std = np.where(std == 0, 1.0, std)
  return (X - media) / std


def procesar_nir_csv(ruta_csv):
  """Devuelve un vector de 24 features (P1..P4 x R..W) o None si hay error."""
  if not os.path.exists(ruta_csv):
    return None
  df = pd.read_csv(ruta_csv)
  # Limpiar posibles espacios en nombres de columnas o puntos
  df.columns = [c.strip() for c in df.columns]
  df["Punto"] = df["Punto"].astype(str).str.strip()

  medias_punto = df.groupby("Punto")[CANALES].mean()
  vector = []
  for p in PUNTOS:
    if p not in medias_punto.index:
      return None
    vector.extend(medias_punto.loc[p, CANALES].values.astype(np.float32))
  return np.array(vector, dtype=np.float32)  # shape (24,)


def construir_tabulares(
    carpeta_nir="muestras_nir",  # cambia a '.' si los CSV están en la raíz
    archivo_brix="muestras_brix/brix_final.xlsx",
    archivo_labels="resultados_granulacion.xlsx",
    total_muestras=100,
    usar_snv=True,
):
  df_brix = pd.read_excel(archivo_brix)
  col_id_brix = df_brix.columns[0]
  df_brix[col_id_brix] = df_brix[col_id_brix].astype(str).str.strip()
  df_brix = df_brix.set_index(col_id_brix)

  df_labels = pd.read_excel(archivo_labels)
  df_labels["ID"] = df_labels["ID"].astype(str).str.strip()
  df_labels = df_labels.set_index("ID")

  ids_validos = []
  X_nir_list = []
  X_brix_list = []
  y_list = []

  for i in range(1, total_muestras + 1):
    muestra_id = f"M{i}"
    ruta_csv = os.path.join(carpeta_nir, f"{muestra_id}.csv")
    # Fallback si el usuario dejó M1.csv en la carpeta actual
    if not os.path.exists(ruta_csv) and os.path.exists(f"{muestra_id}.csv"):
      ruta_csv = f"{muestra_id}.csv"

    vec_nir = procesar_nir_csv(ruta_csv)
    if vec_nir is None:
      print(f"[Aviso] Falta o está incompleto {muestra_id}.csv; se omite.")
      continue

    if muestra_id not in df_brix.index or muestra_id not in df_labels.index:
      print(f"[Aviso] Falta {muestra_id} en Brix o en etiquetas; se omite.")
      continue

    brix_vals = (
        df_brix.loc[muestra_id, PUNTOS].values.astype(np.float32).flatten()
    )
    etiqueta = int(df_labels.loc[muestra_id, "Granulación"])

    ids_validos.append(muestra_id)
    X_nir_list.append(vec_nir)
    X_brix_list.append(brix_vals)
    y_list.append(etiqueta)

  X_nir = np.array(X_nir_list, dtype=np.float32)  # (N, 24)
  X_brix = np.array(X_brix_list, dtype=np.float32)  # (N, 4)
  y = np.array(y_list, dtype=np.int32)  # (N,)
  ids = np.array(ids_validos)

  if usar_snv:
    X_nir = aplicar_snv(X_nir)

  np.save("X_nir_24.npy", X_nir)
  np.save("X_brix_4.npy", X_brix)
  np.save("y_tabulares.npy", y)
  np.save("ids_tabulares.npy", ids)

  print(f"Muestras alineadas: {len(ids)}")
  print(f"X_nir shape: {X_nir.shape} | X_brix shape: {X_brix.shape}")
  print(f"Distribución de clases (0 vs 1): {np.bincount(y)}")


if __name__ == "__main__":
  construir_tabulares(total_muestras=100, usar_snv=True)