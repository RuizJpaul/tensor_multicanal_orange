"""Utilidades compartidas del proyecto.

- Rutas del proyecto (relativas a la raíz con pathlib).
- Carga de cada modalidad VERIFICANDO que los IDs (M1..M100) estén alineados.
- Partición estratificada fija (datos/particiones.npz).
- Semilla global para resultados reproducibles.
- Entrenamiento estándar, selección de umbral en validación y métricas.
- Guardado de métricas/predicciones en resultados/ (sin gráficos).

IMPORTANTE: importar este módulo ANTES que keras/tensorflow para que las
variables de entorno de reproducibilidad tengan efecto.
"""
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # resultados deterministas en CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido en consola

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# RUTAS Y CONSTANTES
# ==========================================
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
MODELOS = RAIZ / "modelos_guardados"
RESULTADOS = RAIZ / "resultados"

SEMILLA = 42
COLS_BRIX = ["P1", "P2", "P3", "P4"]
COLUMNAS_ETIQUETA = [
    "Granulación",
    "Granulacion",
    "Etiqueta (granulación)",
    "Etiqueta",
]


# ==========================================
# REPRODUCIBILIDAD
# ==========================================
def fijar_semilla(semilla: int = SEMILLA) -> None:
  """Fija semillas de Python, NumPy y Keras/TensorFlow."""
  import keras  # import diferido: respeta las variables de entorno de arriba

  os.environ["PYTHONHASHSEED"] = str(semilla)
  random.seed(semilla)
  np.random.seed(semilla)
  keras.utils.set_random_seed(semilla)
  try:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
  except Exception:  # pragma: no cover - depende del backend
    pass


# ==========================================
# CARGA DE DATOS ALINEADA POR ID
# ==========================================
def _normalizar_ids(valores) -> np.ndarray:
  return np.array([str(v).strip().upper() for v in valores])


def _verificar_ids(ids: np.ndarray, ids_ref: np.ndarray, fuente: str) -> None:
  if len(ids) != len(ids_ref) or not np.array_equal(ids, ids_ref):
    difs = [
        (i, a, b) for i, (a, b) in enumerate(zip(ids, ids_ref)) if a != b
    ][:5]
    raise ValueError(
        f"Los IDs de '{fuente}' no están alineados con las etiquetas "
        f"(n={len(ids)} vs {len(ids_ref)}). Primeras diferencias: {difs}"
    )


def _verificar_finito(X: np.ndarray, fuente: str) -> None:
  if not np.isfinite(X).all():
    filas = np.where(~np.isfinite(X).all(axis=tuple(range(1, X.ndim))))[0]
    raise ValueError(f"'{fuente}' tiene valores vacíos/no numéricos en filas {filas[:10]}")


def cargar_etiquetas():
  """Devuelve (ids, y) desde Etiquetas_granulacion_M1_M100.xlsx."""
  ruta = DATOS / "Etiquetas_granulacion_M1_M100.xlsx"
  df = pd.read_excel(ruta)
  df.columns = df.columns.astype(str).str.strip()
  col = next((c for c in COLUMNAS_ETIQUETA if c in df.columns), None)
  if col is None:
    raise KeyError(f"No se encontró columna de etiqueta en {ruta.name}: {list(df.columns)}")
  y = pd.to_numeric(df[col], errors="coerce")
  if y.isna().any():
    raise ValueError(f"Etiquetas vacías o inválidas en {ruta.name}")
  return _normalizar_ids(df["ID"]), y.to_numpy(dtype=np.int32)


def cargar_nir(ids_ref: np.ndarray):
  """Devuelve (X_nir [N, 24], nombres_columnas) alineado con ids_ref."""
  ruta = DATOS / "Dataset_AS7263_M1_M100.xlsx"
  df = pd.read_excel(ruta)
  df.columns = df.columns.astype(str).str.strip()
  _verificar_ids(_normalizar_ids(df["ID"]), ids_ref, ruta.name)
  cols = [c for c in df.columns if c != "ID"]
  X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
  _verificar_finito(X, ruta.name)
  return X, cols


def cargar_brix(ids_ref: np.ndarray):
  """Devuelve (X_brix [N, 4], columnas P1..P4) alineado con ids_ref."""
  ruta = DATOS / "Mediciones_brix.xlsx"
  df = pd.read_excel(ruta)
  df.columns = [
      "ID" if str(c).strip() in ("", "nan", "Unnamed: 0") else str(c).strip()
      for c in df.columns
  ]
  _verificar_ids(_normalizar_ids(df["ID"]), ids_ref, ruta.name)
  X = df[COLS_BRIX].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
  _verificar_finito(X, ruta.name)
  return X, list(COLS_BRIX)


def cargar_imagenes(ids_ref: np.ndarray, y_ref: np.ndarray | None = None) -> np.ndarray:
  """Devuelve el tensor (N, 224, 224, 3) alineado con ids_ref."""
  X = np.load(DATOS / "X_dataset_multicanal.npy").astype(np.float32)
  _verificar_ids(_normalizar_ids(np.load(DATOS / "ids_imagenes.npy")), ids_ref, "ids_imagenes.npy")
  if y_ref is not None:
    y_npy = np.load(DATOS / "y_dataset_etiquetas.npy").astype(np.int32)
    if not np.array_equal(y_npy, y_ref):
      raise ValueError("y_dataset_etiquetas.npy no coincide con el Excel de etiquetas.")
  _verificar_finito(X, "X_dataset_multicanal.npy")
  return X


def cargar_particiones(y: np.ndarray):
  """Carga datos/particiones.npz o la crea (70/15/15 estratificada) si no existe."""
  ruta = DATOS / "particiones.npz"
  if ruta.exists():
    p = np.load(ruta)
    idx_tr, idx_va, idx_te = p["idx_train"], p["idx_val"], p["idx_test"]
  else:
    idx_all = np.arange(len(y))
    idx_tmp, idx_te, y_tmp, _ = train_test_split(
        idx_all, y, test_size=0.15, stratify=y, random_state=SEMILLA
    )
    idx_tr, idx_va = train_test_split(
        idx_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEMILLA
    )
    np.savez(ruta, idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te)
    print(f"[OK] Creada {ruta.name} (70/15/15 estratificada).")

  conjuntos = [set(idx_tr), set(idx_va), set(idx_te)]
  if (conjuntos[0] & conjuntos[1]) or (conjuntos[0] & conjuntos[2]) or (conjuntos[1] & conjuntos[2]):
    raise ValueError("particiones.npz tiene índices repetidos entre train/val/test.")
  if len(conjuntos[0] | conjuntos[1] | conjuntos[2]) != len(y):
    raise ValueError("particiones.npz no cubre todas las muestras.")
  return idx_tr, idx_va, idx_te


# ==========================================
# ENTRENAMIENTO ESTÁNDAR
# ==========================================
def pesos_de_clase(y_train: np.ndarray) -> dict:
  clases = np.unique(y_train)
  pesos = compute_class_weight("balanced", classes=clases, y=y_train)
  return {int(c): float(w) for c, w in zip(clases, pesos)}


def entrenar(modelo, X_tr, y_tr, X_va, y_va, epochs, batch_size=8, patience=10, verbose=0) -> dict:
  """Entrena con pesos de clase (calculados en train) y EarlyStopping en val_loss."""
  import keras

  historial = modelo.fit(
      X_tr,
      y_tr,
      validation_data=(X_va, y_va),
      epochs=epochs,
      batch_size=batch_size,
      class_weight=pesos_de_clase(y_tr),
      callbacks=[
          keras.callbacks.EarlyStopping(
              monitor="val_loss", patience=patience, restore_best_weights=True
          )
      ],
      shuffle=True,
      verbose=verbose,
  )
  return historial.history


# ==========================================
# UMBRAL Y MÉTRICAS
# ==========================================
def elegir_umbral(y_true: np.ndarray, probs: np.ndarray) -> float:
  """Umbral que maximiza F1 (desempate: índice de Youden y cercanía a 0.5).

  Se evalúan puntos medios entre probabilidades consecutivas para no
  quedar "pegado" a una muestra concreta. Usar SOLO con validación.
  """
  if len(np.unique(y_true)) < 2:
    return 0.5
  valores = np.unique(probs)
  candidatos = np.unique(np.concatenate([(valores[:-1] + valores[1:]) / 2, [0.5]]))
  mejor_clave, mejor_t = None, 0.5
  for t in candidatos:
    pred = (probs >= t).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    sens = recall_score(y_true, pred, zero_division=0)
    espec = recall_score(1 - y_true, 1 - pred, zero_division=0)
    clave = (round(f1, 6), round(sens + espec - 1, 6), -abs(t - 0.5))
    if mejor_clave is None or clave > mejor_clave:
      mejor_clave, mejor_t = clave, float(t)
  return mejor_t


def calcular_metricas(y_true: np.ndarray, probs: np.ndarray, umbral: float = 0.5) -> dict:
  pred = (probs >= umbral).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
  return {
      "umbral": float(umbral),
      "n": int(len(y_true)),
      "n_granuladas": int(y_true.sum()),
      "accuracy": float(accuracy_score(y_true, pred)),
      "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
      "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None,
      "f1_granulada": float(f1_score(y_true, pred, zero_division=0)),
      "precision_granulada": float(precision_score(y_true, pred, zero_division=0)),
      "recall_granulada": float(recall_score(y_true, pred, zero_division=0)),
      "especificidad": float(tn / (tn + fp)) if (tn + fp) else 0.0,
      "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
  }


def _fmt(v) -> str:
  return "  n/a " if v is None else f"{v:.4f}"


def imprimir_metricas(titulo: str, m: dict) -> None:
  print(f"--- {titulo} (umbral = {m['umbral']:.3f}) ---")
  print(f"Matriz de confusión [[TN, FP], [FN, TP]]: {m['matriz_confusion']}")
  print(
      f"Accuracy: {_fmt(m['accuracy'])} | Bal. Acc: {_fmt(m['balanced_accuracy'])} | "
      f"ROC-AUC: {_fmt(m['roc_auc'])}"
  )
  print(
      f"Granulada -> F1: {_fmt(m['f1_granulada'])} | Precisión: {_fmt(m['precision_granulada'])} | "
      f"Recall: {_fmt(m['recall_granulada'])} | Especificidad: {_fmt(m['especificidad'])}"
  )


def evaluar_y_guardar(rama, titulo, modelo, X_va, y_va, X_te, y_te, ids_te, historial, extra=None) -> dict:
  """Elige umbral en validación, evalúa test (0.5 y umbral elegido) y guarda resultados."""
  p_va = modelo.predict(X_va, verbose=0).ravel()
  p_te = modelo.predict(X_te, verbose=0).ravel()
  umbral = elegir_umbral(y_va, p_va)

  m_val = calcular_metricas(y_va, p_va, umbral)
  m_te_05 = calcular_metricas(y_te, p_te, 0.5)
  m_te_u = calcular_metricas(y_te, p_te, umbral)

  print(f"\n=== EVALUACIÓN {titulo} (TEST, n={len(y_te)}, granuladas={int(y_te.sum())}) ===")
  imprimir_metricas("Test con umbral fijo 0.50", m_te_05)
  imprimir_metricas("Test con umbral elegido en VALIDACIÓN", m_te_u)
  print(
      classification_report(
          y_te, (p_te >= umbral).astype(int), labels=[0, 1],
          target_names=["Sana (0)", "Granulada (1)"], digits=4, zero_division=0,
      )
  )

  resumen = {
      "rama": rama,
      "semilla": SEMILLA,
      "umbral_validacion": umbral,
      "epocas_entrenadas": len(historial.get("loss", [])),
      "validacion": m_val,
      "test_umbral_050": m_te_05,
      "test_umbral_validacion": m_te_u,
      **(extra or {}),
  }
  RESULTADOS.mkdir(parents=True, exist_ok=True)
  with open(RESULTADOS / f"metricas_{rama}.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
  pd.DataFrame(
      {
          "ID": ids_te,
          "y_real": y_te,
          "prob_granulada": np.round(p_te, 6),
          "pred_umbral_050": (p_te >= 0.5).astype(int),
          "pred_umbral_validacion": (p_te >= umbral).astype(int),
      }
  ).to_csv(RESULTADOS / f"predicciones_test_{rama}.csv", index=False)
  pd.DataFrame(historial).rename_axis("epoca").to_csv(RESULTADOS / f"historial_{rama}.csv")
  print(f"Resultados guardados en {RESULTADOS} (metricas_{rama}.json, predicciones, historial)")
  return resumen
