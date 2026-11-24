"""Rama CNN: red convolucional sobre el tensor multicanal (RGB + UV + NIR)."""
import utilidades_datos as ud  # importar primero (configura TensorFlow)

from arquitecturas import construir_cnn

RAMA = "cnn"


def main():
  ud.fijar_semilla()
  ids, y = ud.cargar_etiquetas()
  X = ud.cargar_imagenes(ids, y_ref=y)
  idx_tr, idx_va, idx_te = ud.cargar_particiones(y)

  modelo = construir_cnn(input_shape=X.shape[1:])
  historial = ud.entrenar(
      modelo, X[idx_tr], y[idx_tr], X[idx_va], y[idx_va], epochs=50, verbose=2
  )
  ud.evaluar_y_guardar(
      RAMA, "RAMA CNN", modelo, X[idx_va], y[idx_va], X[idx_te], y[idx_te], ids[idx_te], historial
  )

  ud.MODELOS.mkdir(parents=True, exist_ok=True)
  modelo.save(ud.MODELOS / "modelo_cnn.keras")
  print(f"Listo: modelo_cnn.keras guardado en {ud.MODELOS}.")


if __name__ == "__main__":
  main()
