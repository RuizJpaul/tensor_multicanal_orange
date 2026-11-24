"""Aplicación web local en Streamlit para el diagnóstico no destructivo de
granulación interna en naranjas Valencia (Citrus sinensis L. Osbeck).

Integra los cuatro modelos entrenados en modelos_guardados/:
  - modelo_cnn.keras (Rama Visual RGB + UV + NIR, 128 dim)
  - modelo_nir.keras + scaler_nir.joblib (Rama Espectral AS7263, 32 dim)
  - modelo_brix.keras + scaler_brix.joblib (Rama Fisicoquímica °Brix, 8 dim)
  - modelo_multimodal.keras + 2 scalers (Fusión Multimodal, 168 dim)

Ejecución local:
  streamlit run app.py
"""
from datetime import datetime
import io
import json
from pathlib import Path
import time

import utilidades_datos as ud  # configura determinismo y rutas

import cv2
import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from estandarizacion.crear_tensor_multicanal import extraer_roi
from estandarizacion.procesar_espectros_nir import (
    CANALES,
    COLUMNAS_ESPECTRALES,
    PUNTOS,
)

RAIZ = ud.RAIZ
DATOS = ud.DATOS
MODELOS = ud.MODELOS
RESULTADOS = ud.RESULTADOS
GRAFICOS = RAIZ / "graficos"

LONGITUDES_NM = [610, 680, 730, 760, 810, 860]
ETIQUETAS_CANALES = ["R (610 nm)", "S (680 nm)", "T (730 nm)", "U (760 nm)", "V (810 nm)", "W (860 nm)"]


# ==========================================
# CARGA EN CACHÉ DE MODELOS Y DATASET
# ==========================================
@st.cache_resource(show_spinner="Cargando redes neuronales (.keras) y escaladores (.joblib) en CPU...")
def cargar_artefactos_produccion():
  m_cnn = keras.models.load_model(MODELOS / "modelo_cnn.keras")
  m_nir = keras.models.load_model(MODELOS / "modelo_nir.keras")
  s_nir = joblib.load(MODELOS / "scaler_nir.joblib")
  m_brix = keras.models.load_model(MODELOS / "modelo_brix.keras")
  s_brix = joblib.load(MODELOS / "scaler_brix.joblib")
  m_mm = keras.models.load_model(MODELOS / "modelo_multimodal.keras")
  s_mm_nir = joblib.load(MODELOS / "scaler_multimodal_nir.joblib")
  s_mm_brix = joblib.load(MODELOS / "scaler_multimodal_brix.joblib")

  umbrales = {}
  for rama in ("cnn", "nir", "brix", "multimodal"):
    ruta_json = RESULTADOS / f"metricas_{rama}.json"
    if ruta_json.exists():
      with open(ruta_json, encoding="utf-8") as f:
        umbrales[rama] = float(json.load(f)["umbral_validacion"])
    else:
      umbrales[rama] = 0.50

  # Calentamiento (warm-up) del grafo en CPU para eliminar latencia de compilación inicial
  _img0 = np.zeros((1, 224, 224, 3), dtype=np.float32)
  _nir0 = np.zeros((1, 24), dtype=np.float32)
  _brix0 = np.zeros((1, 4), dtype=np.float32)
  _ = m_cnn(_img0, training=False)
  _ = m_nir(_nir0, training=False)
  _ = m_brix(_brix0, training=False)
  _ = m_mm([_img0, _nir0, _brix0], training=False)

  return {
      "cnn": m_cnn,
      "nir": m_nir,
      "scaler_nir": s_nir,
      "brix": m_brix,
      "scaler_brix": s_brix,
      "multimodal": m_mm,
      "scaler_mm_nir": s_mm_nir,
      "scaler_mm_brix": s_mm_brix,
      "umbrales": umbrales,
  }


@st.cache_data(show_spinner="Cargando lote experimental M1–M100...")
def cargar_lote_experimental():
  ids, y = ud.cargar_etiquetas()
  X_nir, cols_nir = ud.cargar_nir(ids)
  X_brix, cols_brix = ud.cargar_brix(ids)
  X_img = ud.cargar_imagenes(ids, y_ref=y)
  idx_tr, idx_va, idx_te = ud.cargar_particiones(y)

  df_etq = pd.read_excel(DATOS / "Etiquetas_granulacion_M1_M100.xlsx")
  df_etq.columns = df_etq.columns.astype(str).str.strip()
  confianza = df_etq["CONFIANZA"].astype(str).tolist() if "CONFIANZA" in df_etq.columns else ["Alta"] * len(ids)
  observacion = df_etq["Observacion"].fillna("Sin observaciones").astype(str).tolist() if "Observacion" in df_etq.columns else [""] * len(ids)

  mapa_particion = {}
  for i in idx_tr:
    mapa_particion[int(i)] = "Entrenamiento (69 %)"
  for i in idx_va:
    mapa_particion[int(i)] = "Validación (16 %)"
  for i in idx_te:
    mapa_particion[int(i)] = "Prueba Independiente / Test (15 %)"

  return {
      "ids": ids,
      "y": y,
      "X_nir": X_nir,
      "cols_nir": cols_nir,
      "X_brix": X_brix,
      "cols_brix": cols_brix,
      "X_img": X_img,
      "particion": [mapa_particion.get(i, "Lote") for i in range(len(ids))],
      "confianza": confianza,
      "observacion": observacion,
  }


# ==========================================
# FUNCIONES DE PREPROCESAMIENTO E INFERENCIA
# ==========================================
def construir_tensor_desde_uploads(bytes_rgb: bytes, bytes_uv: bytes, bytes_nir: bytes) -> np.ndarray:
  """Decodifica 3 imágenes cargadas, recorta el ROI adaptativo y normaliza a (224, 224, 3) en [0, 1]."""
  archivos = [(bytes_rgb, False, "RGB"), (bytes_uv, True, "UV"), (bytes_nir, True, "NIR")]
  canales = []
  for raw_bytes, fondo_oscuro, etiqueta in archivos:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError(f"El archivo de imagen {etiqueta} está corrupto o tiene un formato no válido.")
    roi = extraer_roi(img, es_fondo_oscuro=fondo_oscuro)
    resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    canales.append(gray)
  tensor = np.stack(canales, axis=-1)
  if not np.isfinite(tensor).all():
    raise ValueError("El tensor de imagen contiene valores no numéricos.")
  return tensor


def procesar_csv_nir_subido(archivo_csv) -> np.ndarray:
  """Procesa un CSV crudo del sensor AS7263 (50 lecturas x 4 puntos) o una fila de 24 columnas."""
  contenido = archivo_csv.getvalue()
  df = None
  for enc in ("utf-8-sig", "latin1"):
    try:
      df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", encoding=enc)
      break
    except Exception:
      continue
  if df is None or df.empty:
    raise ValueError("No se pudo decodificar el archivo CSV del sensor AS7263.")

  df.columns = df.columns.astype(str).str.strip()
  # Caso A: Formato ya promediado (24 columnas R_P1..W_P4)
  if all(col in df.columns for col in COLUMNAS_ESPECTRALES):
    vec = df[COLUMNAS_ESPECTRALES].iloc[0].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(vec).all():
      raise ValueError("El archivo NIR contiene valores nulos o no numéricos en las 24 variables.")
    return vec

  # Caso B: Formato crudo de AS7263_WEB.html (columnas Punto, R, S, T, U, V, W)
  cols_req = ["Punto"] + CANALES
  faltantes = [c for c in cols_req if c not in df.columns]
  if faltantes:
    raise ValueError(f"El CSV del sensor AS7263 no contiene las columnas requeridas: {faltantes}")

  df = df[cols_req].copy()
  df["Punto"] = df["Punto"].astype(str).str.strip().str.upper()
  df = df[df["Punto"].isin(PUNTOS)].copy()
  for c in CANALES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  df = df.dropna(subset=CANALES)
  if df.empty:
    raise ValueError("No se encontraron mediciones válidas para los puntos P1–P4.")

  promedios = df.groupby("Punto")[CANALES].mean().reindex(PUNTOS)
  if promedios.isna().any().any():
    puntos_falt = list(promedios[promedios.isna().any(axis=1)].index)
    raise ValueError(f"Faltan lecturas en los puntos ecuatoriales: {puntos_falt}")

  valores = [float(promedios.loc[p, c]) for p in PUNTOS for c in CANALES]
  return np.array(valores, dtype=np.float32)


def ejecutar_inferencia_completa(artefactos: dict, tensor_img: np.ndarray, vec_nir: np.ndarray, vec_brix: np.ndarray, usar_umbral_val: bool):
  """Ejecuta las 4 ramas midiendo la latencia real en milisegundos."""
  x_img = np.expand_dims(tensor_img.astype(np.float32), axis=0)
  x_nir = vec_nir.reshape(1, -1).astype(np.float32)
  x_brix = vec_brix.reshape(1, -1).astype(np.float32)

  # Inferencia multimodal cronometrada (RNF-02 / RF-09: < 200 ms)
  t0 = time.perf_counter()
  x_mm_nir = artefactos["scaler_mm_nir"].transform(x_nir)
  x_mm_brix = artefactos["scaler_mm_brix"].transform(x_brix)
  prob_mm = float(np.asarray(artefactos["multimodal"]([x_img, x_mm_nir, x_mm_brix], training=False)).ravel()[0])
  latencia_mm_ms = (time.perf_counter() - t0) * 1000.0

  # Inferencias unimodales comparativas (RF-06)
  prob_cnn = float(np.asarray(artefactos["cnn"](x_img, training=False)).ravel()[0])
  prob_nir = float(np.asarray(artefactos["nir"](artefactos["scaler_nir"].transform(x_nir), training=False)).ravel()[0])
  prob_brix = float(np.asarray(artefactos["brix"](artefactos["scaler_brix"].transform(x_brix), training=False)).ravel()[0])

  probs = {"cnn": prob_cnn, "nir": prob_nir, "brix": prob_brix, "multimodal": prob_mm}
  resultados_ramas = {}
  for rama, p in probs.items():
    u = artefactos["umbrales"][rama] if usar_umbral_val else 0.50
    pred = int(p >= u)
    resultados_ramas[rama] = {
        "prob": p,
        "umbral": u,
        "pred": pred,
        "etiqueta": "Granulada (1)" if pred == 1 else "Sana (0)",
    }

  return resultados_ramas, latencia_mm_ms


def graficar_comparativa_espectral_muestra(vec_nir: np.ndarray, X_nir_lote: np.ndarray, y_lote: np.ndarray, id_muestra: str):
  bandas_lote = X_nir_lote.reshape(-1, 4, 6).mean(axis=1)
  bandas_muestra = vec_nir.reshape(4, 6).mean(axis=0)
  std_muestra = vec_nir.reshape(4, 6).std(axis=0)

  sanas_media = bandas_lote[y_lote == 0].mean(axis=0)
  gran_media = bandas_lote[y_lote == 1].mean(axis=0)

  fig, ax = plt.subplots(figsize=(5.8, 3.6))
  x = np.arange(len(CANALES))
  ax.plot(x, sanas_media, "--", color="#2b5c8f", lw=1.6, alpha=0.8, label="Media Lote Sanas (0)")
  ax.plot(x, gran_media, "--", color="#d95f02", lw=1.6, alpha=0.8, label="Media Lote Granuladas (1)")
  ax.plot(x, bandas_muestra, "-o", color="#111111", lw=2.3, markersize=6, label=f"Muestra {id_muestra} (Media P1–P4)")
  ax.fill_between(x, bandas_muestra - std_muestra, bandas_muestra + std_muestra, color="#7570b3", alpha=0.18, label="±1 DE entre P1–P4")
  ax.set_xticks(x)
  ax.set_xticklabels(ETIQUETAS_CANALES, rotation=22, ha="right", fontsize=8.5)
  ax.set_ylabel("Reflectancia AS7263", fontsize=9)
  ax.set_title(f"Firma Espectral NIR (610–860 nm) — {id_muestra}", fontweight="bold", fontsize=10)
  ax.grid(alpha=0.3, linestyle="--")
  ax.legend(fontsize=7.8)
  plt.tight_layout()
  return fig


# ==========================================
# INTERFAZ PRINCIPAL STREAMLIT
# ==========================================
def main():
  st.set_page_config(
      page_title="Diagnóstico Multimodal — Naranjas Valencia",
      page_icon="🍊",
      layout="wide",
  )

  if "historial_sesion" not in st.session_state:
    st.session_state["historial_sesion"] = []

  artefactos = cargar_artefactos_produccion()
  lote = cargar_lote_experimental()

  # ---------------- BARRA LATERAL ----------------
  with st.sidebar:
    st.markdown("## 🍊 Panel de Control")
    st.caption("Detección No Destructiva de Granulación Interna en Naranjas Valencia")
    st.divider()

    modo = st.radio(
        "Modo de Operación:",
        options=[
            "1. Explorador del Lote Experimental (M1–M100)",
            "2. Diagnóstico de Nueva Muestra (Carga Manual)",
        ],
        index=0,
    )

    criterio_umbral = st.radio(
        "Criterio de Umbral de Decisión:",
        options=[
            "Umbral calibrado en Validación (τ*)",
            "Umbral estándar fijo (0.500)",
        ],
        index=0,
        help="El umbral τ* maximiza el F1-score y el índice J de Youden sobre el conjunto de validación interna.",
    )
    usar_umbral_val = criterio_umbral.startswith("Umbral calibrado")

    st.divider()
    st.markdown("### ✅ Estado de Artefactos")
    st.success(
        "4 modelos `.keras` y 4 escaladores `.joblib` activos en CPU local (`2,67 MB` en memoria)."
    )
    st.markdown(
        f"- **CNN Visual:** τ* = `{artefactos['umbrales']['cnn']:.3f}`\n"
        f"- **MLP Espectral NIR:** τ* = `{artefactos['umbrales']['nir']:.3f}`\n"
        f"- **MLP °Brix:** τ* = `{artefactos['umbrales']['brix']:.3f}`\n"
        f"- **Fusión Multimodal:** τ* = `{artefactos['umbrales']['multimodal']:.3f}`"
    )

  # ---------------- ENCABEZADO ----------------
  st.title("🍊 Sistema Multimodal de Detección de Granulación Interna en Naranjas Valencia")
  st.markdown(
      "**Arquitectura Híbrida de Aprendizaje Profundo (CNN + MLP NIR AS7263 + MLP °Brix — 168 Dimensiones Latentes)**  \n"
      "*Facultad de Ingeniería — Carrera de Ingeniería de Sistemas Computacionales*"
  )

  tab_diag, tab_hist, tab_cv = st.tabs([
      "🔬 1. Diagnóstico Multimodal en Tiempo Real",
      "📋 2. Historial y Exportación de Reportes (RF-10)",
      "📊 3. Auditoría Científica y Validación Cruzada (5×3)",
  ])

  # =========================================================================
  # PESTAÑA 1: DIAGNÓSTICO MULTIMODAL
  # =========================================================================
  with tab_diag:
    tensor_actual = None
    nir_actual = None
    brix_actual = None
    id_actual = None
    meta_real = None

    if modo.startswith("1."):
      col_f1, col_f2, col_f3 = st.columns([1.3, 1.3, 1.4])
      with col_f1:
        filtro_part = st.selectbox(
            "Filtrar por Partición Experimental:",
            options=[
                "Prueba Independiente / Test (15 %)",
                "Todas las muestras (N = 100)",
                "Validación (16 %)",
                "Entrenamiento (69 %)",
            ],
            index=0,
        )
      with col_f2:
        filtro_clase = st.selectbox(
            "Filtrar por Condición de Referencia (Corte):",
            options=["Todas (Sanas y Granuladas)", "Solo Granuladas (Etiqueta = 1)", "Solo Sanas (Etiqueta = 0)"],
            index=0,
        )

      indices_filtrados = []
      for i in range(len(lote["ids"])):
        ok_p = (filtro_part == "Todas las muestras (N = 100)") or (lote["particion"][i] == filtro_part)
        ok_c = (
            filtro_clase.startswith("Todas")
            or (filtro_clase.startswith("Solo Granuladas") and lote["y"][i] == 1)
            or (filtro_clase.startswith("Solo Sanas") and lote["y"][i] == 0)
        )
        if ok_p and ok_c:
          indices_filtrados.append(i)

      if not indices_filtrados:
        st.warning("No hay muestras que coincidan con el filtro seleccionado.")
      else:
        opciones_muestra = [
            f"{lote['ids'][i]} — {'GRANULADA (1)' if lote['y'][i] == 1 else 'SANA (0)'} [{lote['particion'][i].split(' ')[0]}]"
            for i in indices_filtrados
        ]
        with col_f3:
          sel_texto = st.selectbox("Seleccionar Fruto (M1–M100):", options=opciones_muestra, index=0)
        idx_sel = indices_filtrados[opciones_muestra.index(sel_texto)]

        id_actual = str(lote["ids"][idx_sel])
        tensor_actual = lote["X_img"][idx_sel]
        nir_actual = lote["X_nir"][idx_sel]
        brix_actual = lote["X_brix"][idx_sel]
        meta_real = {
            "y_real": int(lote["y"][idx_sel]),
            "particion": lote["particion"][idx_sel],
            "confianza": lote["confianza"][idx_sel],
            "observacion": lote["observacion"][idx_sel],
        }

    else:
      st.markdown("### 📥 Ingreso de Datos de Nueva Muestra Intacta (Validación RNF-04)")
      c_in1, c_in2, c_in3 = st.columns([1.2, 1.2, 1.0])
      with c_in1:
        st.markdown("**1. Imágenes Multiespectrales (RGB, UV, NIR)**")
        id_manual = st.text_input("Código Identificador de Muestra:", value="M_NUEVA_01")
        up_rgb = st.file_uploader("Fotografía Banda Visible (RGB)", type=["jpg", "jpeg", "png"])
        up_uv = st.file_uploader("Fotografía Banda Ultravioleta (UV)", type=["jpg", "jpeg", "png"])
        up_nir = st.file_uploader("Fotografía Banda Infrarroja Cercana (NIR)", type=["jpg", "jpeg", "png"])

      with c_in2:
        st.markdown("**2. Firma Espectral Sensor AS7263 (610–860 nm)**")
        up_csv = st.file_uploader(
            "Archivo CSV exportado de AS7263_WEB.html (50 tomas × 4 puntos)",
            type=["csv"],
        )
        usar_plantilla_nir = st.checkbox("Usar vector espectral de muestra de referencia (M21) para prueba rápida", value=False)

      with c_in3:
        st.markdown("**3. Sólidos Solubles Ecuatoriales (°Brix)**")
        b_p1 = st.number_input("Punto Ecuatorial P1 (°Brix):", min_value=0.0, max_value=25.0, value=9.2, step=0.1)
        b_p2 = st.number_input("Punto Ecuatorial P2 (°Brix):", min_value=0.0, max_value=25.0, value=9.0, step=0.1)
        b_p3 = st.number_input("Punto Ecuatorial P3 (°Brix):", min_value=0.0, max_value=25.0, value=8.8, step=0.1)
        b_p4 = st.number_input("Punto Ecuatorial P4 (°Brix):", min_value=0.0, max_value=25.0, value=9.1, step=0.1)

      if st.button("🚀 Validar Entradas y Ejecutar Diagnóstico Multimodal", type="primary"):
        try:
          if up_rgb and up_uv and up_nir:
            tensor_actual = construir_tensor_desde_uploads(up_rgb.getvalue(), up_uv.getvalue(), up_nir.getvalue())
          elif usar_plantilla_nir:
            idx_m21 = list(lote["ids"]).index("M21")
            tensor_actual = lote["X_img"][idx_m21]
          else:
            raise ValueError("Debe cargar las 3 fotografías (RGB, UV y NIR) o activar la casilla de muestra de prueba.")

          if up_csv is not None:
            nir_actual = procesar_csv_nir_subido(up_csv)
          elif usar_plantilla_nir:
            idx_m21 = list(lote["ids"]).index("M21")
            nir_actual = lote["X_nir"][idx_m21]
          else:
            raise ValueError("Debe cargar el archivo CSV del sensor AS7263 o activar la casilla de muestra de prueba.")

          brix_actual = np.array([b_p1, b_p2, b_p3, b_p4], dtype=np.float32)
          id_actual = id_manual.strip() or "M_NUEVA"
        except Exception as err:
          st.error(f"⚠️ Error de validación de entrada (RNF-04): {err}")

    # ---------------- VISUALIZACIÓN DE ENTRADAS E INFERENCIA ----------------
    if tensor_actual is not None and nir_actual is not None and brix_actual is not None:
      resultados_ramas, latencia_ms = ejecutar_inferencia_completa(
          artefactos, tensor_actual, nir_actual, brix_actual, usar_umbral_val
      )
      res_mm = resultados_ramas["multimodal"]

      st.divider()
      # 1. Tarjeta superior de dictamen y latencia
      c_ver1, c_ver2, c_ver3, c_ver4 = st.columns([1.6, 1.1, 1.1, 1.2])
      with c_ver1:
        if res_mm["pred"] == 1:
          st.error(f"### 🟠 DICTAMEN MULTIMODAL ({id_actual}): GRANULACIÓN INTERNA DETECTADA (1)")
        else:
          st.success(f"### 🟢 DICTAMEN MULTIMODAL ({id_actual}): FRUTO SANO / SIN GRANULACIÓN (0)")
      with c_ver2:
        st.metric(
            "Probabilidad Multimodal",
            f"{res_mm['prob']:.2%}",
            delta=f"Umbral τ = {res_mm['umbral']:.3f}",
            delta_color="off",
        )
      with c_ver3:
        st.metric(
            "Latencia de Inferencia (CPU)",
            f"{latencia_ms:.1f} ms",
            delta="Cumple < 200 ms (RF-09)",
            delta_color="normal",
        )
      with c_ver4:
        if meta_real is not None:
          etq_real_txt = "Granulada (1)" if meta_real["y_real"] == 1 else "Sana (0)"
          coincide = (res_mm["pred"] == meta_real["y_real"])
          st.metric(
              "Referencia Corte Real (Ground Truth)",
              etq_real_txt,
              delta=f"{'✔ Coincide' if coincide else '✖ Discrepancia'} | Confianza: {meta_real['confianza']}",
              delta_color="normal" if coincide else "inverse",
          )
        else:
          st.metric("Modo de Evaluación", "Muestra Externa", delta="Inferencia Local")

      # 2. Comparativa de las 4 arquitecturas lado a lado (RF-06 y RF-07)
      st.markdown("#### 🧠 Desglose Comparativo por Rama Neuronal y Fusión Multimodal")
      cols_m = st.columns(4)
      titulos_ramas = [
          ("cnn", "1. Rama Visual CNN (128 dim)", "Tensor 224×224×3 (RGB+UV+NIR)"),
          ("nir", "2. Rama Espectral NIR (32 dim)", "Sensor AS7263 (24 vars, 610–860 nm)"),
          ("brix", "3. Rama Fisicoquímica (8 dim)", "Refractometría °Brix (P1–P4)"),
          ("multimodal", "4. Fusión Multimodal (168 dim)", "Concatenación Conjunta (CNN+NIR+°Brix)"),
      ]
      for col_ui, (clave, titulo_r, sub_r) in zip(cols_m, titulos_ramas):
        info_r = resultados_ramas[clave]
        with col_ui:
          with st.container(border=True):
            st.markdown(f"**{titulo_r}**")
            st.caption(sub_r)
            st.progress(min(1.0, max(0.0, info_r["prob"])), text=f"Prob: {info_r['prob']:.3f} (τ = {info_r['umbral']:.3f})")
            if info_r["pred"] == 1:
              st.warning(f"Predicción: **{info_r['etiqueta']}**")
            else:
              st.info(f"Predicción: **{info_r['etiqueta']}**")

      # 3. Paneles de inspección de las 3 modalidades de la muestra
      st.markdown("#### 🔍 Inspección Multisensorial de la Muestra Evaluada")
      col_v1, col_v2, col_v3 = st.columns([1.35, 1.35, 1.0])

      with col_v1:
        with st.container(border=True):
          st.markdown("**Modalidad 1: Tensor Visual Multiespectral (`224×224×3`)**")
          fig_img, axes_img = plt.subplots(1, 4, figsize=(7.2, 2.2))
          nombres_c = ["Canal 0: RGB", "Canal 1: UV", "Canal 2: NIR", "Falso Color"]
          for c_i in range(3):
            axes_img[c_i].imshow(tensor_actual[:, :, c_i], cmap="inferno" if c_i > 0 else "gray", vmin=0, vmax=1)
            axes_img[c_i].set_title(nombres_c[c_i], fontsize=8, fontweight="bold")
            axes_img[c_i].axis("off")
          axes_img[3].imshow(np.clip(tensor_actual, 0, 1))
          axes_img[3].set_title(nombres_c[3], fontsize=8, fontweight="bold")
          axes_img[3].axis("off")
          plt.tight_layout()
          st.pyplot(fig_img, use_container_width=True)
          plt.close(fig_img)

          medias_roi = tensor_actual.mean(axis=(0, 1))
          stds_roi = tensor_actual.std(axis=(0, 1))
          st.dataframe(
              pd.DataFrame({
                  "Canal Óptico": ["RGB (Visible)", "UV (Fluorescencia)", "NIR (Infrarrojo Cercano)"],
                  "Media ROI [0-1]": np.round(medias_roi, 4),
                  "Desv. Est. ROI": np.round(stds_roi, 4),
              }),
              use_container_width=True,
              hide_index=True,
          )

      with col_v2:
        with st.container(border=True):
          st.markdown("**Modalidad 2: Espectroscopía NIR Sensor AS7263 (`610–860 nm`)**")
          fig_nir = graficar_comparativa_espectral_muestra(nir_actual, lote["X_nir"], lote["y"], id_actual)
          st.pyplot(fig_nir, use_container_width=True)
          plt.close(fig_nir)

          mat_nir = pd.DataFrame(
              nir_actual.reshape(4, 6),
              index=["Punto P1", "Punto P2", "Punto P3", "Punto P4"],
              columns=["R (610)", "S (680)", "T (730)", "U (760)", "V (810)", "W (860)"],
          ).round(2)
          st.dataframe(mat_nir, use_container_width=True)

      with col_v3:
        with st.container(border=True):
          st.markdown("**Modalidad 3: Sólidos Solubles (`°Brix`)**")
          df_b = pd.DataFrame({
              "Punto Ecuatorial": ["P1 (0°)", "P2 (90°)", "P3 (180°)", "P4 (270°)"],
              "°Brix": np.round(brix_actual, 2),
          })
          st.dataframe(df_b, use_container_width=True, hide_index=True)
          st.markdown(
              f"- **Promedio Ecuatorial:** `{brix_actual.mean():.2f} °Brix`\n"
              f"- **Mínimo Localizado:** `{brix_actual.min():.2f} °Brix`\n"
              f"- **Rango Interno (Máx − Mín):** `{(brix_actual.max() - brix_actual.min()):.2f} °Brix`\n"
              f"- **Desviación Estándar Interna:** `{brix_actual.std():.3f} °Brix`"
          )
          if meta_real is not None and meta_real["observacion"]:
            st.caption(f"**Nota de inspección interna:** {meta_real['observacion']}")

      # Registrar evaluación en el historial de la sesión (RF-10)
      registro = {
          "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "ID_Muestra": id_actual,
          "Particion": meta_real["particion"] if meta_real else "Externa",
          "Etiqueta_Real": meta_real["y_real"] if meta_real else "N/A",
          "Brix_Promedio": round(float(brix_actual.mean()), 3),
          "Brix_Min": round(float(brix_actual.min()), 3),
          "NIR_Media_Global": round(float(nir_actual.mean()), 3),
          "Prob_CNN": round(resultados_ramas["cnn"]["prob"], 4),
          "Prob_NIR": round(resultados_ramas["nir"]["prob"], 4),
          "Prob_Brix": round(resultados_ramas["brix"]["prob"], 4),
          "Prob_Multimodal": round(res_mm["prob"], 4),
          "Umbral_Aplicado": round(res_mm["umbral"], 3),
          "Prediccion_Multimodal": res_mm["pred"],
          "Diagnostico": res_mm["etiqueta"],
          "Latencia_ms": round(latencia_ms, 2),
      }
      if not st.session_state["historial_sesion"] or st.session_state["historial_sesion"][-1]["ID_Muestra"] != id_actual or st.session_state["historial_sesion"][-1]["Umbral_Aplicado"] != registro["Umbral_Aplicado"]:
        st.session_state["historial_sesion"].append(registro)

  # =========================================================================
  # PESTAÑA 2: HISTORIAL Y EXPORTACIÓN DE REPORTES (RF-10)
  # =========================================================================
  with tab_hist:
    st.markdown("### 📋 Registro Auditado de Predicciones y Exportación Tabular (RF-10)")
    col_b1, col_b2, _ = st.columns([1.2, 1.2, 2.0])
    with col_b1:
      if st.button("➕ Evaluar y Agregar las 15 Muestras de Prueba Independiente (Test)"):
        for idx_t in ud.cargar_particiones(lote["y"])[2]:
          i_t = int(idx_t)
          r_ramas, lat_t = ejecutar_inferencia_completa(
              artefactos, lote["X_img"][i_t], lote["X_nir"][i_t], lote["X_brix"][i_t], usar_umbral_val
          )
          st.session_state["historial_sesion"].append({
              "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "ID_Muestra": str(lote["ids"][i_t]),
              "Particion": lote["particion"][i_t],
              "Etiqueta_Real": int(lote["y"][i_t]),
              "Brix_Promedio": round(float(lote["X_brix"][i_t].mean()), 3),
              "Brix_Min": round(float(lote["X_brix"][i_t].min()), 3),
              "NIR_Media_Global": round(float(lote["X_nir"][i_t].mean()), 3),
              "Prob_CNN": round(r_ramas["cnn"]["prob"], 4),
              "Prob_NIR": round(r_ramas["nir"]["prob"], 4),
              "Prob_Brix": round(r_ramas["brix"]["prob"], 4),
              "Prob_Multimodal": round(r_ramas["multimodal"]["prob"], 4),
              "Umbral_Aplicado": round(r_ramas["multimodal"]["umbral"], 3),
              "Prediccion_Multimodal": r_ramas["multimodal"]["pred"],
              "Diagnostico": r_ramas["multimodal"]["etiqueta"],
              "Latencia_ms": round(lat_t, 2),
          })
    with col_b2:
      if st.button("🗑️ Limpiar Historial de Sesión"):
        st.session_state["historial_sesion"] = []

    df_hist = pd.DataFrame(st.session_state["historial_sesion"])
    if not df_hist.empty:
      st.dataframe(df_hist, use_container_width=True, hide_index=True)

      c_dl1, c_dl2, _ = st.columns([1.2, 1.2, 2.0])
      with c_dl1:
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar Reporte de Trazabilidad (.CSV)",
            data=csv_bytes,
            file_name="reporte_diagnostico_multimodal_valencia.csv",
            mime="text/csv",
        )
      with c_dl2:
        buffer_xlsx = io.BytesIO()
        with pd.ExcelWriter(buffer_xlsx, engine="openpyxl") as writer:
          df_hist.to_excel(writer, sheet_name="Diagnosticos_Multimodal", index=False)
        st.download_button(
            "📥 Descargar Reporte de Trazabilidad (.XLSX)",
            data=buffer_xlsx.getvalue(),
            file_name="reporte_diagnostico_multimodal_valencia.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
      st.info("Aún no hay registros en la sesión actual.")

  # =========================================================================
  # PESTAÑA 3: AUDITORÍA CIENTÍFICA Y VALIDACIÓN CRUZADA (5×3)
  # =========================================================================
  with tab_cv:
    st.markdown("### 📊 Resultados Oficiales de Validación Cruzada Estratificada Repetida (5 Folds × 3 Repeticiones, N = 100)")
    ruta_cv = RESULTADOS / "cv_resumen.csv"
    if ruta_cv.exists():
      df_cv = pd.read_csv(ruta_cv)
      st.dataframe(df_cv, use_container_width=True, hide_index=True)

    c_g1, c_g2 = st.columns(2)
    img_comp = GRAFICOS / "graficos_multimodal" / "05_comparativa_global_modelos_cv.png"
    img_roc = GRAFICOS / "graficos_multimodal" / "06_curvas_roc_comparadas_todos_modelos.png"
    with c_g1:
      if img_comp.exists():
        st.image(str(img_comp), caption="Comparativa Global de Modelos en Validación Cruzada 5×3 (N = 100)")
    with c_g2:
      if img_roc.exists():
        st.image(str(img_roc), caption="Curvas ROC Medias Comparadas (Validación Cruzada 5×3, N = 100)")


if __name__ == "__main__":
  main()
