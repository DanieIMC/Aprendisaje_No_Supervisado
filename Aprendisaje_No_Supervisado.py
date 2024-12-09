import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = r"C:\Users\s2dan\OneDrive\Documentos\WorkSpace\Proyect_AI\ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(file_path)

# Codificar las variables categóricas
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Dividir el dataset en características (X), sin utilizar la variable target 'NObeyesdad'
X = data.drop('NObeyesdad', axis=1)  # Características (sin target)

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
# Vamos a probar con un número arbitrario de clusters (por ejemplo, 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Obtener los resultados de los clusters
clusters = kmeans.predict(X_scaled)

# Añadir los resultados de los clusters al dataframe
data['Cluster'] = clusters

# Ver algunas de las filas con los clusters asignados
print(data[['Cluster', 'Age', 'Height', 'Weight', 'Gender', 'family_history_with_overweight']].head())

# Visualizar los resultados usando dos características (por ejemplo, Age y Weight)
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Feature 1 (Age)')  # Reemplazar con el nombre de la característica
plt.ylabel('Feature 2 (Weight)')  # Reemplazar con el nombre de la característica
plt.title('Clustering con K-Means')
plt.colorbar(label='Cluster')
plt.show()

# Ver el número de elementos en cada cluster
print(f"Distribución de los clusters: \n{pd.Series(clusters).value_counts()}")
