import os
import subprocess
import sys
import numpy as np
from sklearn.model_selection import train_test_split


def correr(cmd):
  print(f"\n>>> Ejecutando: {' '.join(cmd)}")
  res = subprocess.run(cmd, check=False)
  if res.returncode != 0:
    print(f"[Error] Falló el comando: {' '.join(cmd)}")
    sys.exit(res.returncode)


def main():
  # 1. Preparar datos tabulares (NIR + Brix) e imágenes (Tensor multicanal)
  correr([sys.executable, "estandarizacion/preparar_tabulares.py"])
  correr([sys.executable, "estandarizacion/tensor_multicanal.py"])

  # 2. Verificar alineación exacta por ID
  ids_tab = np.load("ids_tabulares.npy")
  ids_img = np.load("ids_imagenes.npy")
  if not np.array_equal(ids_tab, ids_img):
    print("[Error] Los IDs de tabulares e imágenes no coinciden en orden.")
    print("Tabulares:", ids_tab)
    print("Imágenes: ", ids_img)
    sys.exit(1)
  print(f"[OK] {len(ids_tab)} muestras alineadas exactamente por ID.")

  # 3. Generar partición estratificada fija (70% train / 15% val / 15% test) si no existe
if not os.path.exists("particiones.npz"):
  y = np.load("y_tabulares.npy")
  idx_all = np.arange(len(y))
  idx_temp, idx_test, y_temp, y_test = train_test_split(
      idx_all, y, test_size=0.15, stratify=y, random_state=42
  )
  idx_train, idx_val = train_test_split(
      idx_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
  )
  np.savez(
      "particiones.npz",
      idx_train=idx_train,
      idx_val=idx_val,
      idx_test=idx_test,
  )
  print("[OK] Creada particiones.npz (70% train / 15% val / 15% test).")

  # 4. Entrenar modelos unimodales y modelo multimodal
  correr([sys.executable, "modelo_nir_mlp.py"])
  correr([sys.executable, "modelo_brix_mlp.py"])
  correr([sys.executable, "modelo_cnn.py"])
  correr([sys.executable, "fusion_multimodal.py"])

  print("\n¡Pipeline completo ejecutado con éxito!")


if __name__ == "__main__":
  main()