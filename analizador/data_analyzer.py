import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import io
import os
import pickle
import joblib
from pathlib import Path
import hashlib

class AnalizadorDataset:
    def __init__(self, dataset_path, dataset_name):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.df = None
        self.X = None
        self.y = None
        self.X_numeric = None
        self.model = None
        self.le = LabelEncoder()
        self._dataset_cargado = False
        self._modelo_entrenado = False
        self._top_10_features = None
        
        # Rutas para guardar modelos (NO dataset)
        self.modelos_dir = Path("modelos_guardados")
        self.modelos_dir.mkdir(exist_ok=True)
        
        # Generar hash único basado en el dataset para el cache
        dataset_hash = self._generar_hash_dataset()
        self.model_path = self.modelos_dir / f"modelo_{dataset_hash}.joblib"
        self.features_path = self.modelos_dir / f"features_{dataset_hash}.pkl"
    
    def _generar_hash_dataset(self):
        """Generar hash único para el dataset"""
        file_path = Path(self.dataset_path) / self.dataset_name
        if file_path.exists():
            # Hash basado en nombre y tamaño del archivo
            file_stat = file_path.stat()
            hash_input = f"{self.dataset_name}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return "default"
    
    def cargar_dataset(self):
        """Cargar dataset SIN CACHE para evitar archivos grandes"""
        if self._dataset_cargado:
            return {
                'success': True,
                'filename': self.dataset_name,
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'tipos_datos': self.df.dtypes.astype(str).to_dict()
            }
            
        try:
            file_path = Path(self.dataset_path) / self.dataset_name
            
            if not file_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo: {self.dataset_name}")
            
            print(f"📖 Cargando dataset desde CSV: {self.dataset_name}")
            self.df = pd.read_csv(file_path)
            self._preparar_datos()
            self._dataset_cargado = True
            
            return {
                'success': True,
                'filename': self.dataset_name,
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'tipos_datos': self.df.dtypes.astype(str).to_dict(),
                'desde_cache': False
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _preparar_datos(self):
        """Preparar datos para análisis - OPTIMIZADO"""
        if self.df.shape[1] < 2:
            raise ValueError("El dataset debe tener al menos 2 columnas (características + target)")
        
        # Última columna como target, resto como características
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        
        # Codificar target si es categórico
        if self.y.dtype == 'object':
            self.y = self.le.fit_transform(self.y)
        
        # Solo usar columnas numéricas para características
        self.X_numeric = self.X.select_dtypes(include=[np.number])
        
        # Si no hay columnas numéricas, intentar convertir
        if self.X_numeric.shape[1] == 0:
            print("⚠️ No hay columnas numéricas, intentando conversión...")
            # Convertir solo primeras columnas para eficiencia
            max_cols = min(50, len(self.X.columns))
            cols_to_convert = self.X.columns[:max_cols]
            self.X_numeric = self.X[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    def _entrenar_o_cargar_modelo(self):
        """Entrenar modelo nuevo o cargar existente"""
        if self._modelo_entrenado:
            return True
            
        # Intentar cargar modelo existente
        if self.model_path.exists() and self.features_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                with open(self.features_path, 'rb') as f:
                    self._top_10_features = pickle.load(f)
                self._modelo_entrenado = True
                print("✅ Modelo de importancia cargado desde archivo")
                return True
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}. Entrenando nuevo...")
        
        # Entrenar nuevo modelo
        if self.X_numeric.shape[1] == 0:
            return False
            
        print("🔄 Entrenando nuevo modelo de importancia...")
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10  # Limitar profundidad para eficiencia
        )
        self.model.fit(self.X_numeric, self.y)
        
        # Calcular importancia
        importancia_df = pd.DataFrame({
            'caracteristica': self.X_numeric.columns,
            'importancia': self.model.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        self._top_10_features = importancia_df.head(10)['caracteristica'].tolist()
        self._modelo_entrenado = True
        
        # Guardar modelo y features (archivos pequeños)
        joblib.dump(self.model, self.model_path, compress=3)
        with open(self.features_path, 'wb') as f:
            pickle.dump(self._top_10_features, f)
        print("💾 Modelo de importancia guardado en archivo")
        
        return True
    
    def visualizar_dataset(self):
        """Mostrar información del dataset sin gráficos"""
        try:
            if not self._dataset_cargado:
                self.cargar_dataset()
            
            # Información básica del dataset
            info = {
                'filas': self.df.shape[0],
                'columnas': self.df.shape[1],
                'nombres_columnas': self.df.columns.tolist(),
                'tipos_datos': self.df.dtypes.astype(str).to_dict(),
                'valores_nulos': self.df.isnull().sum().to_dict(),
                'primeras_10_filas': self.df.head(10).to_dict('records')
            }
            
            # DataFrame describe() como diccionario
            describe_info = {}
            numeric_cols = self.df.select_dtypes(include=[np.number])
            if not numeric_cols.empty:
                # Solo calcular describe para columnas numéricas
                describe_info = numeric_cols.describe().to_dict()
            
            # Información general del dataset
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_string = buffer.getvalue()
            
            return {
                'success': True,
                'info': info,
                'describe': describe_info,
                'info_string': info_string
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def importancia_caracteristicas(self):
        """Calcular importancia de características"""
        try:
            if not self._dataset_cargado:
                self.cargar_dataset()
                
            if not self._entrenar_o_cargar_modelo():
                return {'success': False, 'error': 'No hay características numéricas para analizar'}
            
            # Obtener importancia del modelo (cargado o entrenado)
            importancia_df = pd.DataFrame({
                'caracteristica': self.X_numeric.columns,
                'importancia': self.model.feature_importances_
            }).sort_values('importancia', ascending=False)
            
            top_10_features = importancia_df.head(10)
            
            modelo_cargado = self.model_path.exists()
            
            return {
                'success': True,
                'importancia_total': importancia_df.to_dict('records'),
                'top_10_caracteristicas': top_10_features.to_dict('records'),
                'columnas_top_10': self._top_10_features,
                'modelo_cargado': modelo_cargado
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def reducir_caracteristicas(self):
        """Reducir características usando modelo guardado"""
        try:
            if not self._dataset_cargado:
                self.cargar_dataset()
                
            if not self._entrenar_o_cargar_modelo():
                return {'success': False, 'error': 'No hay características numéricas'}
            
            # Usar las características ya calculadas
            X_reduced = self.X_numeric[self._top_10_features]
            
            return {
                'success': True,
                'caracteristicas_originales': self.X_numeric.shape[1],
                'caracteristicas_reducidas': len(self._top_10_features),
                'caracteristicas_seleccionadas': self._top_10_features,
                'caracteristicas_eliminadas': [col for col in self.X_numeric.columns if col not in self._top_10_features],
                'reduccion_porcentaje': f"{(1 - len(self._top_10_features) / self.X_numeric.shape[1]) * 100:.2f}%",
                'nuevo_shape': X_reduced.shape
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calcular_f1_score(self):
        """Calcular F1 Score con modelo rápido"""
        try:
            if not self._dataset_cargado:
                self.cargar_dataset()
                
            if not self._entrenar_o_cargar_modelo():
                return {'success': False, 'error': 'No hay características numéricas'}
            
            X_top_10 = self.X_numeric[self._top_10_features]
            
            # Ruta para modelo de F1 score
            f1_model_path = self.modelos_dir / f"f1_model_{self._generar_hash_dataset()}.joblib"
            
            # Intentar cargar modelo F1 existente
            if f1_model_path.exists():
                try:
                    f1_model = joblib.load(f1_model_path)
                    print("✅ Modelo F1 cargado desde archivo")
                    modelo_cargado = True
                except Exception as e:
                    print(f"⚠️ Error cargando modelo F1: {e}. Entrenando nuevo...")
                    modelo_cargado = False
            else:
                modelo_cargado = False
            
            # Si no existe o hay error, entrenar nuevo
            if not modelo_cargado:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_top_10, self.y, test_size=0.3, random_state=42, stratify=self.y
                )
                
                f1_model = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=8  # Más rápido
                )
                f1_model.fit(X_train, y_train)
                joblib.dump(f1_model, f1_model_path, compress=3)
                print("💾 Modelo F1 guardado en archivo")
            
            # Usar datos de prueba para métricas
            X_train, X_test, y_train, y_test = train_test_split(
                X_top_10, self.y, test_size=0.3, random_state=42, stratify=self.y
            )
            
            if not modelo_cargado:
                # Ya tenemos el modelo recién entrenado
                y_pred = f1_model.predict(X_test)
            else:
                # Usar modelo cargado
                y_pred = f1_model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = f1_model.score(X_test, y_test)
            
            return {
                'success': True,
                'f1_score': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'caracteristicas_utilizadas': len(self._top_10_features),
                'caracteristicas_lista': self._top_10_features,
                'muestras_entrenamiento': X_train.shape[0],
                'muestras_prueba': X_test.shape[0],
                'modelo_cargado': modelo_cargado
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def limpiar_cache(self):
        """Limpiar archivos de modelos guardados"""
        try:
            files_eliminados = 0
            for file in self.modelos_dir.glob("*"):
                try:
                    file.unlink()
                    files_eliminados += 1
                    print(f"🗑️ Eliminado: {file.name}")
                except Exception as e:
                    print(f"❌ Error eliminando {file.name}: {e}")
            
            self._dataset_cargado = False
            self._modelo_entrenado = False
            self._top_10_features = None
            self.model = None
            
            print(f"✅ Cache limpiado. Archivos eliminados: {files_eliminados}")
            return True
        except Exception as e:
            print(f"❌ Error limpiando cache: {e}")
            return False