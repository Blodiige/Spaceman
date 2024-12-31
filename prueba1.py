import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# import json

# # Paso 1: Leer el archivo .txt
# with open('message.txt', 'r') as file:
#     data = file.read()

# # Paso 2: Cargar los datos JSON
# try:
#     json_data = json.loads(data)  # Cargar el contenido JSON en una variable
# except json.JSONDecodeError as e:
#     print(f"Error al leer el archivo JSON: {e}")
#     json_data = []

# # Paso 3: Extraer los valores de 'gameResult' y agregar a una lista
# game_results = []
# for entry in json_data:
#     game_result = entry.get('gameResult')
#     if game_result is not None:  # Solo agregar si 'gameResult' no es None
#         game_results.append(float(game_result))  # Convertir a flotante si es necesario

# Paso 4: Imprimir los resultados
# Paso 1: Generar 500 números aleatorios con 2 decimales entre 1 y 1000
numeros_aleatorios =[3.8, 1.08, 1.24, 1.59, 6.45, 2.88, 1.59, 1.66, 1.0, 4.39, 1.19, 1.61, 3.23, 1.0, 5.12, 1.09, 4.63, 2.28, 1.03, 1.44, 2.46, 1.15, 1.0, 1.67, 1.3, 3.43, 1.23, 1.33, 3.35, 1.37, 3.35, 1.79, 1.26, 1.02, 2.48, 1.2, 17.54, 1.39, 2.6, 2.98, 1.38, 6.62, 1.41, 3.22, 2.06, 5.18, 13.92, 4.83, 4.54, 1.92, 1.11, 1.0, 4.59, 1.48, 1.19, 1.32, 1.0, 2.98, 2.35, 1.17, 2.63, 1.03, 3.25, 1.08, 1.81, 2.78, 3.06, 1.14, 25.43, 6.67, 13.31, 1.45, 5.91, 1.18, 1.16, 1.92, 9.44, 1.36, 6.36, 1.31, 4.39, 2.46, 1.0, 1.97, 1.64, 1.0, 55.23, 1.62, 2.18, 2.32, 1.46, 1.27, 2.27, 6.47, 1142.8, 1.14, 2.85, 1.07, 5.77, 7.48, 1.02, 1.27, 6.16, 4.77, 1.24, 1.0, 1.24, 1.39, 29.51, 12.95, 2.61, 1.0, 3.57, 31.69, 1.0, 1.47, 1.07, 2.35, 1.09, 1.88, 2.2, 1.87, 4.0, 1.25, 1.92, 9.38, 2.16, 1.18, 4.25, 1.61, 1.06, 1.92, 3.38, 2.55, 1.0, 10.98, 1.0, 1.86, 1.91, 5.15, 1.12, 1.61, 1.37, 2.03, 1.0, 25.77, 1.09, 33.55, 1.63, 4.07, 1.4, 3.2, 11.66, 2.08, 10.48, 1.18, 25.17, 4.5, 1.32, 1.47, 1.25, 1.26, 2.7, 2.22, 2.05, 1.0, 2.21, 5.49, 23.17, 2.3, 2.36, 20.92, 8.26, 1.14, 1.0, 1.6, 4.27, 1.17, 1.12, 7.8, 5.99, 4.37, 1.89, 14.07, 1.09, 3.02, 1.23, 1.64, 3.13, 1.3, 16.86, 1.02, 6.57, 1.06, 1.14, 1.32, 1.82, 5.9, 1.92, 1.0, 1.49, 1.09, 3.33, 2.86, 1.21, 8.89, 8.17, 2.02, 1.07, 121.31, 1.0, 4.07, 2.21, 1.15, 4.15, 2.17, 1.63, 2.37, 1.79, 5.67, 1.06, 3.49, 2.17, 2.43, 1.39, 2.33, 1.49, 5.73, 1.44, 1.7, 1.02, 1.59, 3.32, 1.34, 3.9, 1.25, 3.27, 2.3, 1.03, 2.46, 1.23, 7.29, 23.71, 1.11, 1.0, 1.26, 1.37, 1.62, 7.92, 1.9, 1.06, 3.16, 2.19, 3.94, 1.14, 2.57, 1.36, 1.83, 20.14, 3.04, 2.88, 1.67, 2.56, 1.17, 1.16, 444.46, 1.05, 1.0, 4.88, 7.29, 1.26, 1.12, 1.41, 13.0, 1.17, 2.05, 1.13, 5.92, 1.51, 9.42, 25.1, 1.33, 1.33, 2.88, 1.09, 2.82, 1.49, 109.01, 10.66, 1.07, 1.02, 2.65, 1.25, 2.05, 20.77, 6.32, 736.46, 1.08, 2.66, 1.53, 1.82, 1.05, 1.2, 2.78, 1.8, 3.62, 1.47, 3.95, 19.38, 2.11, 1.81, 1.88, 2.36, 4.98, 7.44, 3.6, 1.44, 1.58, 1.5, 2.38, 1.03, 2.51, 170.59, 1.45, 1.14, 3.31, 7.36, 9.02, 2.53, 1.56, 2.54, 1.68, 4.06, 4.79, 1.37, 1.93, 1.23, 1.13, 45.98, 1.39, 1.79, 1.17, 1.37, 1.28, 1.03, 8.41, 1.0, 2.47, 3.34, 1.23, 1.06, 4.47, 2.73, 3.71, 1.68, 2.59, 1.41, 1.09, 2.2, 2.2, 4.24, 2.07, 41.23, 16.35, 1047.11, 2.26, 1.01, 1.22, 4.73, 1.07, 2.84, 8.08, 3.09, 1.0, 3.16, 1.72, 1.26, 1.23, 1.79, 2.3, 1.43, 1.89, 1.65, 67.05, 1.14, 11.27, 616.82, 5.56, 1.35, 1.1, 28.27, 1.23, 10.41, 1.29, 1.23, 3.75, 1.62, 1.7, 7.08, 1.43, 1.72, 1.32, 1.23, 1.23, 6.53, 1.0, 1.18, 2.98, 6.78, 2.01, 2.73, 1.37, 2.11, 1.04, 54.85, 1.24, 1.27, 1.26, 1.75, 1.47, 1.0, 1.18, 1.43, 3.49, 1.58, 4.35, 1.11, 1.24, 5.89, 1.3, 1.75, 4.13, 1.37, 2.33, 1.46, 1.47, 3.14, 1.43, 2.33, 6.78, 3.72, 1.0, 1.0, 1.13, 1.25, 1.28, 1.08, 1.11, 1.0, 1.93, 1.64, 1.93, 1.15, 5.54, 2.07, 2.01, 9.13, 1.72, 1.05, 3.28, 2.85, 1.08, 1.28, 3.05, 4.62, 3.22, 2.31, 1.07, 1.02, 1.06, 2.36, 38.68, 2.1, 1.47, 1.03, 1.1, 11.22, 1.7, 4.47, 2.92, 1.34, 12.67, 1.05, 5.62, 1.83, 1.39, 1.97, 1.6, 3.14, 1.38, 1.67, 1.94, 9.34, 1.56, 1.0, 1.06, 1.79, 2.03, 6.89, 1.64]
# Paso 2: Preparar los datos para el modelo LSTM
n_steps = 30  # Número de pasos en la secuencia
X, y = [], []

# Crear las secuencias y objetivos
for i in range(len(numeros_aleatorios) - n_steps):
    X.append(numeros_aleatorios[i:i + n_steps])
    y.append(numeros_aleatorios[i + n_steps])

X, y = np.array(X), np.array(y)

# Remodelar para que LSTM lo acepte como entrada
X = X.reshape((X.shape[0], X.shape[1], 1))

# Paso 3: Definir el modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Paso 4: Entrenar el modelo
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Paso 5: Proceso iterativo con nueva entrada
while True:
    # Predecir el siguiente número
    ultima_secuencia = np.array(numeros_aleatorios[-n_steps:]).reshape(1, n_steps, 1)
    prediccion = model.predict(ultima_secuencia)
    
    print(f"Última secuencia: {numeros_aleatorios[-n_steps:]}")
    print(f"Predicción del siguiente número: {prediccion[0][0]:.2f}")
    
    # Solicitar el número real al usuario
    numero_real = float(input("Ingrese el número real: "))
    error = abs(prediccion[0][0] - numero_real)
    print(f"Error de predicción: {error:.2f}")
    
    # Actualizar los datos con el nuevo número real
    numeros_aleatorios.append(numero_real)
    
    # Volver a generar X y y con los datos actualizados
    X, y = [], []
    for i in range(len(numeros_aleatorios) - n_steps):
        X.append(numeros_aleatorios[i:i + n_steps])
        y.append(numeros_aleatorios[i + n_steps])

    X, y = np.array(X), np.array(y)

    # Remodelar para que LSTM lo acepte como entrada
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Reentrenar el modelo con los datos actualizados
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    print("\nModelo reentrenado con el nuevo dato.")
