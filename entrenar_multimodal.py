"""Modelo multimodal: fusión de imagen multicanal + espectro NIR + °Brix."""
import utilidades_datos as ud  # importar primero (configura TensorFlow)

import joblib
from sklearn.preprocessing import StandardScaler

from arquitecturas import construir_multimodal

RAMA = "multimodal"


def main():
  ud.fijar_semilla()
  ids, y = ud.cargar_etiquetas()
  X_img = ud.cargar_imagenes(ids, y_ref=y)
  X_nir, _ = ud.cargar_nir(ids)
  X_brix, _ = ud.cargar_brix(ids)
  idx_tr, idx_va, idx_te = ud.cargar_particiones(y)

  # Escalar SOLO con train
  scaler_nir = StandardScaler().fit(X_nir[idx_tr])
  scaler_brix = StandardScaler().fit(X_brix[idx_tr])

  def entradas(idx):
    return [X_img[idx], scaler_nir.transform(X_nir[idx]), scaler_brix.transform(X_brix[idx])]

  modelo = construir_multimodal(X_img.shape[1:], X_nir.shape[1], X_brix.shape[1])
  historial = ud.entrenar(
      modelo, entradas(idx_tr), y[idx_tr], entradas(idx_va), y[idx_va], epochs=80, verbose=2
  )
  ud.evaluar_y_guardar(
      RAMA, "MODELO MULTIMODAL", modelo,
      entradas(idx_va), y[idx_va], entradas(idx_te), y[idx_te], ids[idx_te], historial,
  )

  ud.MODELOS.mkdir(parents=True, exist_ok=True)
  modelo.save(ud.MODELOS / "modelo_multimodal.keras")
  joblib.dump(scaler_nir, ud.MODELOS / "scaler_multimodal_nir.joblib")
  joblib.dump(scaler_brix, ud.MODELOS / "scaler_multimodal_brix.joblib")
  print(f"Listo: modelo_multimodal.keras y escaladores guardados en {ud.MODELOS}.")


if __name__ == "__main__":
  main()
