#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This script designs a basic XGboost model to predict the objective variable HIT. 
# I will start a process to search for relationships by loading both tables. 

# Este script disena un modelo XGboost basico para predecir la variable objetiva HIT 
# Comenzare un proceso para buscar relaciones cargando ambas tablas  

import pandas as pd

# File paths
# Rutas de los archivos
path_biogrid = "C:/Polimeromics/data/exported_data/biogrid_homosapiens.csv"
path_rcsb = "C:/Polimeromics/data/exported_data/rcsb_pdb.csv"

# Load tables
# Cargar las tablas
biogrid = pd.read_csv(path_biogrid)
rcsb = pd.read_csv(path_rcsb)

# Preview of the tables
# Vista previa de las tablas
print("Biogrid Homo Sapiens:")
print(biogrid.head())

print("\nRCSB PDB:")
print(rcsb.head())


# In[2]:


# Ensure that tables are limited to records related to Homo sapiens:
#  Asegurarse que las tablas estén limitadas a los registros relacionados con Homo sapiens: 

# Filter tables
# Filtrar las tablas
biogrid_filtered = biogrid[biogrid["organism_official"] == "Homo sapiens"]
rcsb_filtered = rcsb[rcsb["taxonomy_id"] == 9606]


# In[3]:


# Create explicit copies when filtering
# Crear copias explícitas al filtrar
biogrid_filtered = biogrid[biogrid["organism_official"] == "Homo sapiens"].copy()
rcsb_filtered = rcsb[rcsb["taxonomy_id"] == 9606].copy()

# Normalize columns
# Normalizar columnas
biogrid_filtered["official_symbol"] = biogrid_filtered["official_symbol"].str.lower().str.strip()
rcsb_filtered["macromolecule_name"] = rcsb_filtered["macromolecule_name"].str.lower().str.strip()


# In[4]:


# Verify 
# Verifico 

print(biogrid_filtered["official_symbol"].head())
print(rcsb_filtered["macromolecule_name"].head())


# In[5]:


# Combine tables based on the standardized columns
# Combinar tablas basándose en las columnas normalizadas
combined_data = biogrid_filtered.merge(
    rcsb_filtered,
    left_on="official_symbol",
    right_on="macromolecule_name",
    how="inner"
)
# Check the number of combined rows
# Verificar el número de filas combinadas
print("Número de filas combinadas:", combined_data.shape[0])

# Show a sample of the combined data
# Mostrar una muestra de los datos combinados
print(combined_data.head())


# In[6]:


# We enrich the data by creating new columns useful for the model:
# Enriquecemos los datos creando nuevas columnas útiles para el modelo:

# Create new features
# Crear nuevas características
combined_data["score_product"] = combined_data["score_1"] * combined_data["score_2"]
combined_data["aliases_length"] = combined_data["aliases"].str.len()
combined_data["ligand_mw_normalized"] = combined_data["ligand_mw"] / combined_data["molecular_weight"]

# Verify the new features
# Verificar las nuevas características
print(combined_data[["score_product", "aliases_length", "ligand_mw_normalized"]].head())


# In[7]:


# We select the relevant characteristics and define the target variable hit as 1 for “YES” and 0 for “NO”:
# Seleccionamos las características relevantes y definimos la variable objetivo hit como 1 para "YES" y 0 para "NO":

# Selection of characteristics and target variable
# Selección de características y variable objetivo
X = combined_data[[
    "score_1", "score_2", "score_product", "aliases_length", "ligand_mw_normalized"
]]
y = combined_data["hit"].apply(lambda x: 1 if x == "YES" else 0)

# Check dimensions
# Verificar dimensiones
print("Tamaño de X:", X.shape)
print("Tamaño de y:", y.shape)


# In[8]:


print(y.value_counts())


# In[9]:


# I apply a balancing technique
# Aplico una tecnica de balanceo

from imblearn.over_sampling import RandomOverSampler

# Oversampling
# Sobremuestreo
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)

# Verify distribution of balanced classes
# Verificar distribución de clases balanceadas
print("Distribución después del sobremuestreo:", y_balanced.value_counts())


# In[10]:


# We split the data into training and test sets to evaluate model performance:
# Dividimos los datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo:

from sklearn.model_selection import train_test_split

# Split balanced data
# Dividir datos balanceados
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Verify sizes of assemblies
# Verificar tamaños de los conjuntos
print("Tamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Distribución de y_train:", y_train.value_counts())
print("Distribución de y_test:", y_test.value_counts())


# In[11]:


# We train the model using XGBoost, an efficient and robust algorithm:
# Entrenamos el modelo utilizando XGBoost, un algoritmo eficiente y robusto:

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Create and train the model
# Crear y entrenar el modelo
model = XGBClassifier(random_state=42,  eval_metric="logloss")
model.fit(X_train, y_train)

# Making predictions on the test set
# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluating performance
# Evaluar el rendimiento
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Confusion matrix
# Matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))


# In[12]:


# Adjusted hyperparameters
# Hiperparámetros ajustados
model = XGBClassifier(
    max_depth=3,  # Reducir complejidad del árbol
    min_child_weight=5,  # Evitar hojas con pocos datos
    reg_lambda=1,  # Regularización L2
    reg_alpha=1,  # Regularización L1
    random_state=42,
    eval_metric="logloss"
)

# Retraining
# Reentrenar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Reassess performance
# Revaluar desempeño
from sklearn.metrics import classification_report, confusion_matrix
print("Reporte de clasificación ajustado:")
print(classification_report(y_test, y_pred))

# Save the trained model for XGBoost
# Guardar el modelo entrenado para XGBoost
import joblib
joblib.dump(model, "C:/Polimeromics/models/xgboost_model.pkl")


# In[13]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Custom cross validation
# Validación cruzada personalizada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X_balanced, y_balanced):
    X_train_cv, X_val_cv = X_balanced.iloc[train_idx], X_balanced.iloc[val_idx]
    y_train_cv, y_val_cv = y_balanced.iloc[train_idx], y_balanced.iloc[val_idx]
    
    # Create a new model for each iteration
    # Crear un nuevo modelo para cada iteración
    model_cv = XGBClassifier(
        max_depth=3,
        min_child_weight=5,
        reg_lambda=1,
        reg_alpha=1,
        random_state=42,
        eval_metric="logloss"
    )
    model_cv.fit(X_train_cv, y_train_cv)
    y_val_pred = model_cv.predict(X_val_cv)
    
    # Calculate F1-Score for this iteration
    # Calcular F1-Score para esta iteración
    scores.append(f1_score(y_val_cv, y_val_pred, average='weighted'))

# Average F1-Score
# Promedio de F1-Score
print("F1-score promedio en validación cruzada:", sum(scores) / len(scores))

