"""Orquestador del pipeline completo.

Uso:
  python run_pipeline.py                          # entrena las 4 ramas + prueba de carga
  python run_pipeline.py --con-preprocesamiento   # regenera además Excel NIR y tensor de imágenes
  python run_pipeline.py --con-cv                 # además corre la validación cruzada (lenta)
"""
import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def ejecutar(script_rel: str, *args: str) -> None:
  print("\n" + "=" * 50)
  print(f">>> Ejecutando: {script_rel} {' '.join(args)}")
  print("=" * 50, flush=True)
  res = subprocess.run([sys.executable, str(RAIZ / script_rel), *args], cwd=str(RAIZ), check=False)
  if res.returncode != 0:
    print(f"\n[Error] Falló {script_rel} (código {res.returncode})")
    sys.exit(res.returncode)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--con-preprocesamiento", action="store_true",
                      help="Regenera Dataset_AS7263_M1_M100.xlsx y X_dataset_multicanal.npy")
  parser.add_argument("--con-cv", action="store_true", help="Ejecuta evaluar_cv_estratificada.py al final")
  args = parser.parse_args()

  if args.con_preprocesamiento:
    ejecutar("estandarizacion/procesar_espectros_nir.py")
    ejecutar("estandarizacion/crear_tensor_multicanal.py")

  ejecutar("entrenar_rama_nir.py")
  ejecutar("entrenar_rama_brix.py")
  ejecutar("entrenar_rama_cnn.py")
  ejecutar("entrenar_multimodal.py")
  ejecutar("test_cargar_modelos.py")

  if args.con_cv:
    ejecutar("evaluar_cv_estratificada.py")

  print("\n" + "=" * 50)
  print(">>> ¡Pipeline completo ejecutado con éxito!")
  print(f">>> Modelos: {RAIZ / 'modelos_guardados'} | Métricas: {RAIZ / 'resultados'}")
  print("=" * 50)


if __name__ == "__main__":
  main()
