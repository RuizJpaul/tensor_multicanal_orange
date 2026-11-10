import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

# 1. Cargar los datasets ya procesados y guardados en formato .npy
print("Cargando los datasets multicanal...")
X_train = np.load('X_dataset_multicanal.npy')
y_train = np.load('y_dataset_etiquetas.npy')

print(f"X_train shape (Lote, Alto, Ancho, Canales): {X_train.shape}")  # Esperado: (50, 224, 224, 3)
print(f"y_train shape (Etiquetas): {y_train.shape}")                    # Esperado: (50,)

# 2. Definir la arquitectura de la Red Convolucional (CNN) Multicanal
def construir_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Clasificador Fully Connected
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Regularización para mitigar sobreajuste
        layers.Dense(1, activation='sigmoid') # Salida binaria (0 = Sano, 1 = Dañado/Granulación)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Instanciar y mostrar resumen del modelo
modelo = construir_cnn()
modelo.summary()

# 3. Entrenar el modelo con las 50 muestras
print("\nIniciando el entrenamiento de la CNN...")
history = modelo.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8,
    validation_split=0.2 # 20% reservado para validación interna del lote
)

# 4. Guardar el modelo entrenado
modelo.save('modelo_granulacion_cnn.h5')
print("\n¡Modelo entrenado y guardado con éxito como 'modelo_granulacion_cnn.h5'!")