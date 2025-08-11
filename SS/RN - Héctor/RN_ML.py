# Predecir la salida para varios entradas, que jamas haya visto el modelo.
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class RedNeuronalHector:
    def __init__(self, hidden_layer_sizes=(100, 50, 20), max_iter=80000, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',           # Función de activación ReLU
            solver='adam',              # Optimizador Adam
            alpha=0.001,               # Regularización L2
            learning_rate_init=0.001,   # Tasa de aprendizaje inicial
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,       # Desactivado para datasets pequeños
            tol=1e-6                   # Tolerancia para convergencia
        )
        
        # Escaladores para normalizar los datos
        self.scaler_X = StandardScaler()  # Para las variables de entrada
        self.scaler_y = StandardScaler()  # Para las variables de salida
        
        self.is_trained = False
        self.training_history = {}
    
    def cargar_datos(self, archivo_csv):
        print("Cargando datos desde CSV...")
        
        # Leer el CSV
        df = pd.read_csv(archivo_csv, sep=' ')
        
        print(f"Datos cargados: {len(df)} registros")
        print(f"Columnas: {list(df.columns)}")
        
        # Separar variables de entrada (X) y salida (y)
        # Entradas: time, risk, arrival
        X = df[['time', 'risk', 'arrival']].values
        
        # Salidas: valencia, arousal
        y = df[['valencia', 'arousal']].values
        
        print(f"\nForma de X (entradas): {X.shape}")
        print(f"Forma de y (salidas): {y.shape}")
        
        return X, y, df
    
    def preparar_datos(self, X, y, test_size=0.2):
        print(f"\nPreparando datos (test_size={test_size})...")
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalizar las variables de entrada
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Normalizar las variables de salida
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        print(f"Datos de entrenamiento: {X_train_scaled.shape[0]} muestras")
        print(f"Datos de prueba: {X_test_scaled.shape[0]} muestras")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_train, X_test, y_train, y_test
    
    def entrenar(self, X_train, y_train):

        print("\nIniciando entrenamiento de la Red Neuronal...")
        
        # Entrenar el modelo
        self.model.fit(X_train, y_train)
        
        # Guardar información del entrenamiento
        self.training_history = {
            'loss_curve': self.model.loss_curve_,
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_
        }
        
        self.is_trained = True
        
        print(f"Entrenamiento completado!")
        print(f"Iteraciones: {self.model.n_iter_}")
        print(f"Pérdida final: {self.model.best_loss_:.6f}")
    
    def predecir(self, X):
        """
        Realiza predicciones con el modelo entrenado
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero!")
        
        # Realizar predicciones 
        y_pred_scaled = self.model.predict(X)
        
        # Desnormalizar las predicciones
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def evaluar(self, X_test, y_test_real):
        """
        Evalúa el rendimiento del modelo
        """
        print("\nEvaluando modelo...")
        
        # Realizar predicciones
        y_pred = self.predecir(X_test)
        
        # Calcular métricas para cada variable de salida
        resultados = {}
        variables = ['Valencia', 'Arousal']
        
        for i, var in enumerate(variables):
            mse = mean_squared_error(y_test_real[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_real[:, i], y_pred[:, i])
            
            resultados[var] = {
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
            
            print(f"\n{var}:")
            print(f"   MSE:  {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²:   {r2:.4f}")
        
        return resultados, y_pred
    
    def predecir_nuevo(self, time, risk, arrival):
        """
        Predice valencia y arousal para nuevos valores de entrada
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero!")
        
        # Crear array con los nuevos datos
        X_nuevo = np.array([[time, risk, arrival]])
        
        # Normalizar
        X_nuevo_scaled = self.scaler_X.transform(X_nuevo)
        
        # Predecir
        prediccion = self.predecir(X_nuevo_scaled)
        
        valencia_pred = prediccion[0, 0]
        arousal_pred = prediccion[0, 1]
        
        return valencia_pred, arousal_pred
    
    def mostrar_arquitectura(self):
        """
        Muestra información sobre la arquitectura de la red
        """
        if not self.is_trained:
            print("El modelo debe ser entrenado primero!")
            return
        
        print("\nARQUITECTURA DE LA RED NEURONAL TIPO 1")
        print("=" * 50)
        print(f"🔹 Capas ocultas: {self.model.hidden_layer_sizes}")
        print(f"🔹 Número total de capas: {self.model.n_layers_}")
        print(f"🔹 Función de activación: {self.model.activation}")
        print(f"🔹 Optimizador: {self.model.solver}")
        print(f"🔹 Variables de entrada: 3 (time, risk, arrival)")
        print(f"🔹 Variables de salida: 2 (valencia, arousal)")
        
        # Mostrar pesos de cada capa
        print(f"\nESTRUCTURA DETALLADA:")
        for i, coef in enumerate(self.model.coefs_):
            if i == 0:
                print(f"   Capa Entrada → Oculta 1: {coef.shape[0]} → {coef.shape[1]} neuronas")
            elif i == len(self.model.coefs_) - 1:
                print(f"   Capa Oculta {i} → Salida: {coef.shape[0]} → {coef.shape[1]} neuronas")
            else:
                print(f"   Capa Oculta {i} → Oculta {i+1}: {coef.shape[0]} → {coef.shape[1]} neuronas")

def ejemplo_uso():
    """
    Ejemplo completo de uso de la Red Neuronal 
    """
    print("RED NEURONAL")
    print("=" * 60)
    
    # Crear instancia de la red neuronal
    red_hector = RedNeuronalHector(
        hidden_layer_sizes=(10, 10),  
        max_iter=3000,
        random_state=42
    )
    
    # Cargó  los datos porque la ultima vez de me olvido xd
    X, y, df = red_hector.cargar_datos('mhernandez.csv')
    
    print("\nUsando todos los datos para entrenamiento...")
    
    # Normalizar todos los datos
    X_scaled = red_hector.scaler_X.fit_transform(X)
    y_scaled = red_hector.scaler_y.fit_transform(y)
    
    # Entrenar la red con todos los datos
    red_hector.entrenar(X_scaled, y_scaled)
    
    # Muestro la arquitectura
    red_hector.mostrar_arquitectura()
    
    # Evaluar el modelo con los mismos datos (solo para demostración)
    print("\nEvaluando modelo con datos de entrenamiento...")
    resultados, predicciones = red_hector.evaluar(X_scaled, y)
    
    # Mostrar todos los datos vs predicciones
    print(f"\n📋 COMPARACIÓN: TODOS LOS DATOS")
    print("=" * 55)
    print(f"{'':>15} {'Valencia':>10} {'Arousal':>10} {'Valencia':>10} {'Arousal':>10}")
    print(f"{'':>15} {'Real':>10} {'Real':>10} {'Pred':>10} {'Pred':>10}")
    print("-" * 55)
    
    for i in range(len(y)):
        print(f"Registro {i+1:>6}: {y[i,0]:>8.1f} {y[i,1]:>8.1f} {predicciones[i,0]:>8.1f} {predicciones[i,1]:>8.1f}")
    
    # Ejemplo de predicción para nuevos datos
    print("\nREDICCIÓN PARA NUEVOS DATOS")
    print("=" * 40)
    
    # Ejemplo: time=0.5, risk=0.7, arrival=0.25
    valencia_pred, arousal_pred = red_hector.predecir_nuevo(0.5, 0.7, 0.25)
    print(f"Para time=0.5, risk=0.7, arrival=0.25:")
    print(f"   Valencia predicha: {valencia_pred:.2f}")
    print(f"   Arousal predicho: {arousal_pred:.2f}")
    
    return red_hector

if __name__ == "_main_":
    modelo_hector = ejemplo_uso()
