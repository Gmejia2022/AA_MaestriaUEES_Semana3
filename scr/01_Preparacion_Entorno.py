"""
01 - Preparacion del Entorno
Proyecto: Taller Colaborativo Semana 3, aprendizaje no supervisado (K-means, DBSCAN, PCA y t-SNE)
Maestria en IA - UEES - Semana 3
Grupo #2
Alumnos: 
Ingeniero Gonzalo Mejia Alcivar
Ingeniero Jorge Ortiz Merchan
Ingeniero David Perugachi Rojas
"""

# === Importacion de librerias ===

# Manejo de datos
import pandas as pd
import numpy as np

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Reduccion de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Utilidades
import os
import warnings

warnings.filterwarnings('ignore')

# === Rutas del proyecto ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def verificar_entorno():
    """Verifica que todas las librerias esten instaladas correctamente."""
    librerias = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'seaborn': sns.__version__,
        'scikit-learn': __import__('sklearn').__version__,
    }

    lineas = []
    lineas.append("=" * 50)
    lineas.append("  VERIFICACION DEL ENTORNO")
    lineas.append("=" * 50)
    for lib, version in librerias.items():
        lineas.append(f"  {lib:20s} -> v{version}")
    lineas.append("=" * 50)
    lineas.append("  Todas las librerias instaladas correctamente.")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def verificar_estructura_proyecto():
    """Verifica que la estructura de carpetas del proyecto exista."""
    carpetas = ['Data', 'Models', 'notebooks', 'results', 'scr']

    lineas = []
    lineas.append("\n  ESTRUCTURA DEL PROYECTO")
    lineas.append("=" * 50)
    for carpeta in carpetas:
        ruta = os.path.join(BASE_DIR, carpeta)
        existe = os.path.exists(ruta)
        estado = "OK" if existe else "NO ENCONTRADA"
        lineas.append(f"  {carpeta:20s} -> {estado}")
    lineas.append("=" * 50)

    texto = "\n".join(lineas)
    print(texto)
    return texto


def cargar_datos():
    """Carga el dataset y muestra informacion basica."""
    ruta_datos = os.path.join(BASE_DIR, 'Data', 'E-commerce Dataset.csv')

    if not os.path.exists(ruta_datos):
        print(f"  ERROR: No se encontro el archivo en {ruta_datos}")
        return None, ""

    df = pd.read_csv(ruta_datos)

    lineas = []
    lineas.append("\n  INFORMACION DEL DATASET")
    lineas.append("=" * 50)
    lineas.append(f"  Filas:    {df.shape[0]}")
    lineas.append(f"  Columnas: {df.shape[1]}")
    lineas.append(f"  Nombres:  {list(df.columns)}")
    lineas.append("=" * 50)

    lineas.append("\n  PRIMERAS 5 FILAS")
    lineas.append("-" * 50)
    lineas.append(df.head().to_string(index=False))

    lineas.append("\n  TIPOS DE DATOS")
    lineas.append("-" * 50)
    for col in df.columns:
        lineas.append(f"  {col:30s} -> {df[col].dtype}")

    lineas.append("\n  ESTADISTICAS DESCRIPTIVAS")
    lineas.append("-" * 50)
    lineas.append(df.describe().to_string())

    lineas.append("\n  VALORES NULOS")
    lineas.append("-" * 50)
    lineas.append(f"  Total valores nulos: {df.isnull().sum().sum()}")

    texto = "\n".join(lineas)
    print(texto)
    return df, texto


def guardar_reporte(txt_entorno, txt_estructura, txt_datos):
    """Guarda el reporte completo de la preparacion del entorno."""
    reporte = []
    reporte.append("ETAPA 1: PREPARACION DEL ENTORNO")
    reporte.append("=" * 50)
    reporte.append(txt_entorno)
    reporte.append(txt_estructura)
    reporte.append(txt_datos)

    ruta_reporte = os.path.join(RESULTS_DIR, '01_preparacion_entorno_reporte.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    print(f"\n  Reporte guardado en: {ruta_reporte}")


if __name__ == '__main__':
    txt_entorno = verificar_entorno()
    txt_estructura = verificar_estructura_proyecto()
    df, txt_datos = cargar_datos()
    guardar_reporte(txt_entorno, txt_estructura, txt_datos)
