from pathlib import Path
import re
import pandas as pd

# ==========================================
# CONFIGURACIÓN DE RUTAS Y VARIABLES
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_CSV = RAIZ / "datos" / "cvs_nir"
ARCHIVO_SALIDA = RAIZ / "datos" / "Dataset_AS7263_M1_M100.xlsx"

CANALES = ["R", "S", "T", "U", "V", "W"]
PUNTOS = ["P1", "P2", "P3", "P4"]
COLUMNAS_ESPECTRALES = [
    f"{canal}_{punto}" for punto in PUNTOS for canal in CANALES
]


# ==========================================
# LECTURA Y LIMPIEZA DE UN ARCHIVO CSV
# ==========================================


def leer_csv(archivo: Path) -> pd.DataFrame:
  for encoding in ("utf-8-sig", "latin1"):
    try:
      return pd.read_csv(archivo, sep=None, engine="python", encoding=encoding)
    except UnicodeDecodeError:
      continue
  raise ValueError(f"No se pudo decodificar {archivo.name}")


def procesar_naranja(archivo: Path):
  identificador = archivo.stem.upper()
  df = leer_csv(archivo)
  df.columns = df.columns.astype(str).str.strip()

  columnas_req = ["Punto"] + CANALES
  faltantes = [col for col in columnas_req if col not in df.columns]
  if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

  df = df[columnas_req].copy()
  df["Punto"] = df["Punto"].astype(str).str.strip().str.upper()
  df = df[df["Punto"].isin(PUNTOS)].copy()

  for canal in CANALES:
    df[canal] = pd.to_numeric(df[canal], errors="coerce")

  # Elimina tramas seriales corruptas o filas incompletas antes de promediar y contar
  df = df.dropna(subset=CANALES)
  if df.empty:
    raise ValueError("No se encontraron mediciones válidas P1-P4")

  promedios = df.groupby("Punto")[CANALES].mean().reindex(PUNTOS)

  fila = {"ID": identificador}
  for punto in PUNTOS:
    for canal in CANALES:
      fila[f"{canal}_{punto}"] = promedios.loc[punto, canal]

  conteo = (
      df.groupby("Punto")
      .size()
      .reindex(PUNTOS, fill_value=0)
      .reset_index(name="Cantidad_mediciones")
  )
  conteo.insert(0, "ID", identificador)

  return fila, conteo


# ==========================================
# PROCESAMIENTO EN LOTE (M1 - M100)
# ==========================================


def procesar_archivos():
  archivos = sorted(
      [
          f
          for f in CARPETA_CSV.glob("*.csv")
          if re.fullmatch(r"M\d+", f.stem, re.IGNORECASE)
      ],
      key=lambda f: int(f.stem[1:]),
  )

  if not archivos:
    print(f"No se encontraron archivos M1.csv a M100.csv en {CARPETA_CSV}.")
    return

  resultados, controles, errores = [], [], []

  for archivo in archivos:
    try:
      fila, conteo = procesar_naranja(archivo)
      resultados.append(fila)
      controles.append(conteo)
    except Exception as e:
      errores.append(f"{archivo.name}: {e}")
      print(f"ERROR en {archivo.name}: {e}")

  if not resultados:
    print("No se pudo procesar ningún archivo.")
    return

  dataset = pd.DataFrame(resultados)[["ID"] + COLUMNAS_ESPECTRALES]
  dataset[COLUMNAS_ESPECTRALES] = dataset[COLUMNAS_ESPECTRALES].round(4)

  control = pd.concat(controles, ignore_index=True)

  ARCHIVO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
  with pd.ExcelWriter(ARCHIVO_SALIDA, engine="openpyxl") as writer:
    dataset.to_excel(writer, sheet_name="Dataset", index=False)
    control.to_excel(writer, sheet_name="Control_mediciones", index=False)

    hoja = writer.sheets["Dataset"]
    hoja.freeze_panes = "B2"
    hoja.auto_filter.ref = hoja.dimensions
    hoja.column_dimensions["A"].width = 12
    for col in range(2, hoja.max_column + 1):
      hoja.column_dimensions[hoja.cell(1, col).column_letter].width = 14

    hoja_ctrl = writer.sheets["Control_mediciones"]
    hoja_ctrl.freeze_panes = "A2"
    hoja_ctrl.auto_filter.ref = hoja_ctrl.dimensions

  print("\n===================================")
  print("PROCESAMIENTO FINALIZADO")
  print("===================================")
  print(f"Archivos procesados: {len(dataset)} / {len(archivos)}")
  print(f"Variables espectrales: {len(COLUMNAS_ESPECTRALES)}")
  print(f"Archivo generado: {ARCHIVO_SALIDA}")

  if errores:
    print("\nARCHIVOS CON ERRORES:")
    for err in errores:
      print(err)


if __name__ == "__main__":
  procesar_archivos()