# Agregar al inicio del archivo
import os
from pathlib import Path

# En la clase AnalizadorDataset, modificar el __init__:
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
    
    # Rutas para guardar modelos - compatible con Render
    if os.environ.get('RENDER'):
        # En Render, usar /tmp para archivos temporales
        self.modelos_dir = Path("/tmp/modelos_guardados")
    else:
        # En desarrollo, usar carpeta local
        self.modelos_dir = Path("modelos_guardados")
    
    self.modelos_dir.mkdir(exist_ok=True)
    
    # Generar hash único basado en el dataset para el cache
    dataset_hash = self._generar_hash_dataset()
    self.model_path = self.modelos_dir / f"modelo_{dataset_hash}.joblib"
    self.features_path = self.modelos_dir / f"features_{dataset_hash}.pkl"
