"""Rama NIR: MLP sobre las 24 variables espectrales del sensor AS7263."""
import utilidades_datos as ud  # importar primero (configura TensorFlow)

import joblib
from sklearn.preprocessing import StandardScaler

from arquitecturas import construir_mlp_nir

RAMA = "nir"


def main():
  ud.fijar_semilla()
  ids, y = ud.cargar_etiquetas()
  X, _ = ud.cargar_nir(ids)
  idx_tr, idx_va, idx_te = ud.cargar_particiones(y)

  # Escalar SOLO con train
  scaler = StandardScaler().fit(X[idx_tr])
  X_tr, X_va, X_te = (scaler.transform(X[i]) for i in (idx_tr, idx_va, idx_te))

  modelo = construir_mlp_nir(X.shape[1])
  historial = ud.entrenar(modelo, X_tr, y[idx_tr], X_va, y[idx_va], epochs=80)
  ud.evaluar_y_guardar(
      RAMA, "RAMA NIR", modelo, X_va, y[idx_va], X_te, y[idx_te], ids[idx_te], historial
  )

  ud.MODELOS.mkdir(parents=True, exist_ok=True)
  modelo.save(ud.MODELOS / "modelo_nir.keras")
  joblib.dump(scaler, ud.MODELOS / "scaler_nir.joblib")
  print(f"Listo: modelo_nir.keras y scaler_nir.joblib guardados en {ud.MODELOS}.")


if __name__ == "__main__":
  main()
