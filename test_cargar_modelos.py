"""Prueba de carga: verifica que cada modelo + su escalador funcionen con los
datos REALES de test y que reproduzcan las métricas guardadas en resultados/.

Uso: python test_cargar_modelos.py
"""
import utilidades_datos as ud  # importar primero (configura TensorFlow)

import json
import sys

import joblib
import keras
import numpy as np


def _cargar_umbral(rama):
  ruta = ud.RESULTADOS / f"metricas_{rama}.json"
  if not ruta.exists():
    return 0.5, None
  with open(ruta, encoding="utf-8") as f:
    info = json.load(f)
  return info["umbral_validacion"], info


def main():
  ids, y = ud.cargar_etiquetas()
  X_nir, _ = ud.cargar_nir(ids)
  X_brix, _ = ud.cargar_brix(ids)
  X_img = ud.cargar_imagenes(ids, y_ref=y)
  _, _, idx_te = ud.cargar_particiones(y)
  y_te = y[idx_te]
  M = ud.MODELOS
  print(f"[OK] Datos alineados por ID. Test: n={len(idx_te)}, granuladas={int(y_te.sum())}\n")

  casos = {
      "nir": lambda: keras.models.load_model(M / "modelo_nir.keras").predict(
          joblib.load(M / "scaler_nir.joblib").transform(X_nir[idx_te]), verbose=0),
      "brix": lambda: keras.models.load_model(M / "modelo_brix.keras").predict(
          joblib.load(M / "scaler_brix.joblib").transform(X_brix[idx_te]), verbose=0),
      "cnn": lambda: keras.models.load_model(M / "modelo_cnn.keras").predict(X_img[idx_te], verbose=0),
      "multimodal": lambda: keras.models.load_model(M / "modelo_multimodal.keras").predict(
          [X_img[idx_te],
           joblib.load(M / "scaler_multimodal_nir.joblib").transform(X_nir[idx_te]),
           joblib.load(M / "scaler_multimodal_brix.joblib").transform(X_brix[idx_te])], verbose=0),
  }

  errores = 0
  print(f"{'rama':<11} {'umbral':>7} {'AUC':>7} {'F1 gran.':>9} {'Recall':>7} {'Acc':>7}  coincide con resultados/")
  for rama, predecir in casos.items():
    try:
      p = predecir().ravel()
      umbral, info = _cargar_umbral(rama)
      m = ud.calcular_metricas(y_te, p, umbral)
      if info is None:
        estado = "(sin metricas_*.json)"
      else:
        ref = info["test_umbral_validacion"]
        ok = np.isclose(m["roc_auc"], ref["roc_auc"], atol=1e-4) and m["matriz_confusion"] == ref["matriz_confusion"]
        estado = "SI" if ok else f"NO (guardado AUC={ref['roc_auc']:.4f})"
        errores += 0 if ok else 1
      print(f"{rama:<11} {umbral:>7.3f} {m['roc_auc']:>7.4f} {m['f1_granulada']:>9.4f} "
            f"{m['recall_granulada']:>7.4f} {m['accuracy']:>7.4f}  {estado}")
    except Exception as e:  # noqa: BLE001
      errores += 1
      print(f"{rama:<11} ERROR: {e}")

  print("\nResultado:", "TODO OK" if errores == 0 else f"{errores} problema(s) detectado(s)")
  sys.exit(1 if errores else 0)


if __name__ == "__main__":
  main()
