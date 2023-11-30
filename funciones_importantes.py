
#original = pd.read_csv('.csv')
#df = original.copy()
#df.shape

#pd.set_option('display.float_format', lambda x: '%.3f' % x)


#df = df.drop(columns=['Unnamed: 0'])

#VALIDAR DATOS DE LAS COLUMNAS
import pandas as pd

def validate_columns(df):
    # Initialize an empty list to store validation results
    validation_results = []

    # Loop through each column in the input DataFrame
    for col in df.columns:
        # Calculate the number of unique values
        num_unique_values = df[col].nunique()

        # Calculate the number of null values
        num_null_values = df[col].isnull().sum()

        # Calculate the percentage of null values
        percent_null_values = (num_null_values / len(df)) * 100

        # Get sample unique values
        sample_unique_values = df[col].dropna().sample(min(num_unique_values, 5)).tolist()

        # Create a dictionary with the validation results for the current column
        validation_results.append({
            'Column': col,
            'Unique_Values': df[col].unique(),
            'Num_Unique_Values': num_unique_values,
            'Num_Null_Values': num_null_values,
            'Sample_Unique_Values': sample_unique_values,
            '%_null' : percent_null_values
        })

    # Convert the list of dictionaries to a DataFrame
    validation_df = pd.DataFrame(validation_results)

    return validation_df



#LEER CSVs
import pandas as pd
import os

def normalize_name(filename):
    return filename.replace("olist_", "").replace("_dataset", "").replace(".csv", "")
def load_all_data(path):
    ''' read all datasets in folder and usea as name'''
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    data = {normalize_name(filename): pd.read_csv(f"{path}/{filename}") for filename in files}
    return data

import matplotlib.pyplot as plt 
import seaborn as sns 



#HACER LOS BOXPLOT
def grafico_out_boxplot(df):
    cols = df.select_dtypes(include=['number']).columns

    filas = len(cols)
    columnas = 1

    fig, axes = plt.subplots(filas, columnas, figsize=(6,50 ))

    for i, columna in enumerate(cols):
        sns.boxplot(y=df[columna], ax=axes[i])
        axes[i].set_title(columna)

    plt.tight_layout()
    return (plt.show())

import seaborn as sns
import matplotlib.pyplot as plt



#HACER LOS HISTOGRAMAS
def plot_histograms(dataframe):
    # Obtén la lista de nombres de todas las columnas del DataFrame
    column_names = dataframe.columns

    # Itera a través de los nombres de las columnas y crea un histograma para cada una
    for column_name in column_names:
        plt.figure(figsize=(8, 6))  # Tamaño de la figura
        sns.histplot(dataframe[column_name], kde=True, edgecolor='w')
        plt.title(f'Histograma de {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frecuencia')
        plt.show()



#T TEST
'''

df='df que uso'
from scipy.stats import ttest_ind
group1 = df[df["first_child"]==True]

group1 = group1[[ 'actual_duration']]

group2 = df[df["first_child"]==False]
group2 = group2[[ 'actual_duration']]

t_stat, p_value = ttest_ind(group1, group2)

if p_value < 0.05:
  print("Rechazar la hipótesis nula. Hay una diferencia significativa de duracion de semanas entre madres primerizas y no primerizas.")
else:
  print("No rechazar la hipótesis nula.") '''


'''
#LATITUD Y LONGITUD
from math import radians, sin, cos, asin, sqrt 
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computa distancia entre dos pares (lat, lng)
    Ver - (https://en.wikipedia.org/wiki/Haversine_formula)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

import pandas as pd
import numpy as np
import os



#PASAR A DTIME
def transformar_columnas_datetime(dataframe, columns_to_transform):
    for column in columns_to_transform:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column])
    return dataframe




def tiempo_de_espera(dataframe, is_delivered):
    # filtrar por entregados y crea la varialbe tiempo de espera
    if is_delivered:
        dataframe = dataframe.query("order_status=='delivered'").copy()
    # compute wait time
    dataframe.loc[:, 'tiempo_de_espera'] = \
        (dataframe['order_delivered_customer_date'] -
         dataframe['order_purchase_timestamp']) / np.timedelta64(24, 'h')
    return dataframe

def tiempo_de_espera_previsto(dataframe, is_delivered):
    # filtrar por entregados y crea la varialbe tiempo de espera previsto
    if is_delivered:
        dataframe = dataframe.query("order_status=='delivered'").copy()
    # compute wait time
    dataframe.loc[:, 'tiempo_de_espera_previsto'] = \
        (dataframe['order_estimated_delivery_date'] -
         dataframe['order_approved_at']) / np.timedelta64(24, 'h')
    return dataframe


def real_vs_esperado(dataframe, is_delivered=True):
    #filtrar por entregados y crea la varialbe tiempre real vs esperado
    if is_delivered:
        dataframe = dataframe.query("order_status == 'delivered'").copy()
    # compute wait time
    dataframe.loc[:, 'real_vs_esperado'] = \
        (dataframe['tiempo_de_espera'] -
         dataframe['tiempo_de_espera_previsto']) 
    # hago que me de 0 si es menor la diferencia 
    dataframe['real_vs_esperado'] = np.where(dataframe['real_vs_esperado'] < 0, 0, dataframe['real_vs_esperado'])
    return dataframe

def puntaje_de_compra(df):
    def es_cinco_estrellas(col): #creo la primer funcion que hace que si es 5 estrellas me de 1, sino 0
        return 1 if col == 5 else 0 

    def es_una_estrella(col): #creo la segunda funcion que hace que si es 1 estrella me de 1, sino 0
        return 1 if col == 1 else 0

    puntajes = pd.DataFrame()
    puntajes['order_id'] = df['order_id'] 
    puntajes['es_cinco_estrellas'] = df['review_score'].apply(es_cinco_estrellas) #aplico primera funcion
    puntajes['es_una_estrella'] = df['review_score'].apply(es_una_estrella) #aplico segunda funcion
    puntajes['review_score'] = df['review_score'] 

    return puntajes

def calcular_numero_productos(dataframe):
        conteo_por_order_id = dataframe.groupby('order_id')['order_item_id'].count() #hago el groupby para que me junte por order_id y me cuente el order_item_id
        conteo_por_order_id = conteo_por_order_id.reset_index()  #lo paso a df (le saco la estructura del groupby)
        conteo_por_order_id = conteo_por_order_id.rename(columns={'order_item_id': 'number_of_products'}) #nombres de las columnas 
        return conteo_por_order_id


def vendedores_unicos(dataframe):
        conteo_por_vendedor_id = dataframe.groupby('order_id')['seller_id'].nunique() #hago el groupby para que me junte por order_id y me traiga los unicos seller_id
        conteo_por_vendedor_id = conteo_por_vendedor_id.reset_index() #hago el df
        conteo_por_vendedor_id = conteo_por_vendedor_id.rename(columns={'order_item_id': 'vendedores_unicos'}) #nombres de columnas
        return conteo_por_vendedor_id

def calcular_precio_y_transporte(dataframe):
     precio_y_transporte_por_orden = dataframe.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index() #hago un groupby por order_id, luego con agg hago calculos con price y freight (sumo), lo paso a df con reset_index
     return precio_y_transporte_por_orden



def calcular_distancia_vendedor_comprador(data): #abro todas los archivos que tengo dentro de data, los abro como copias como al principio
    for df in data:
        if df == 'order_items':
            order_items = data[df].copy()
        elif df == 'orders':
            orders = data[df].copy()
        elif df == 'sellers':
            sellers = data[df].copy()
        elif df == 'customers':
            customers = data[df].copy()
        elif df == 'geolocation':
            geolocation = data[df].copy()
            geo = geolocation.groupby('geolocation_zip_code_prefix').first() #hago el groupby con first(), que lo que hace es traerme el primero 

    orders_customers = orders[['order_id', 'customer_id']] #traigo estas columnas
    df_customers = pd.merge(orders_customers, customers,how= 'inner', on='customer_id') #hago el merge por customer_id (lo que tienen en comun)
    df_customers = df_customers.drop('customer_city', axis=1)
    df_customers = df_customers.drop('customer_state', axis=1)
    df_customers = df_customers.rename(columns={'customer_zip_code_prefix':'geolocation_zip_code_prefix'}) #nombres de columnas
    df_customers = pd.merge(df_customers, geo, how='inner', on='geolocation_zip_code_prefix') #hago el merge
    df_customers = df_customers.rename(columns={'geolocation_lat': 'lat_customer', 'geolocation_lng':'lng_customer'}) #nombres de columnas
    df_sellers = pd.merge(orders_customers, order_items, how='inner', on='order_id')
    columnas_excluir = ['order_item_id','product_id','shipping_limit_date','price', 'freight_value']
    df_sellers = df_sellers.drop(columnas_excluir, axis=1) #elimino lo que no uso 
    df_sellers = pd.merge(df_sellers, sellers, how='inner', on='seller_id') #hago el merge de sellers por seller_id
    columnas_excluir = ['seller_city', 'seller_state']
    df_sellers = df_sellers.drop(columnas_excluir, axis=1)
    df_sellers = df_sellers.rename(columns={'seller_zip_code_prefix':'geolocation_zip_code_prefix'})
    df_sellers = pd.merge(df_sellers, geo, how='inner', on='geolocation_zip_code_prefix')
    columnas_excluir = ['customer_id', 'geolocation_city','geolocation_state']
    df_sellers = df_sellers.drop(columnas_excluir, axis=1)
    df_sellers = df_sellers.rename(columns={'geolocation_lat': 'lat_seller', 'geolocation_lng':'lng_seller'})
    df_final = pd.merge(df_customers, df_sellers, how='inner', on='order_id')
    df_final = df_final.dropna() #borro vacios y configuro tipos
    df_final['lat_seller'] = pd.to_numeric(df_final['lat_seller'], errors = 'coerce')
    df_final['lng_seller'] = pd.to_numeric(df_final['lng_seller'], errors = 'coerce')
    df_final['lat_customer'] = pd.to_numeric(df_final['lat_customer'], errors = 'coerce')
    df_final['lng_customer'] = pd.to_numeric(df_final['lng_customer'], errors = 'coerce')
    
    distances = []
    for index, row in df_final.iterrows():
        distance = haversine_distance(row['lng_customer'], row['lat_customer'], row['lng_seller'], row['lat_seller'])
        distances.append(distance)
    df_final['distancia_a_la_orden'] = distances


    return df_final[['order_id','distancia_a_la_orden']]

'''
#CORRELACION HEATMAP
'''
numeric_columns = df.select_dtypes(include=[np.number])  # Selecciona solo las columnas numéricas

plt.figure(figsize = (10, 10))

sns.heatmap(numeric_columns.corr(), cmap = 'coolwarm', annot = True, annot_kws = {"size": 10})



#COMO HACER UN MODELO univariable
import statsmodels.formula.api as smf

model1 = smf.ols(formula='quiero predecir ~ variable que quiero ver si afecta',data=df)
model1=model1.fit()
model1.summary()

#como estandarizo
from sklearn.preprocessing import StandardScaler

features = ['','', '','', '', '', '']
scaler = StandardScaler()

# estandarizo lo que le mande en la variable y lo fiteo
df[features] = scaler.fit_transform(df[features])

#COMO HAGO UN MODELO CON +1 VARIABLE
formula = "review_score ~ " + ' + '.join(features)
model = smf.ols(formula,data=df)
model=model.fit()
model.summary()
print(f" El modelo dio de r-cuadrado {model.rsquared}")


#GRAFICAR LOS INTERCEPTOS DEL MODELO EN UN GRAFICO DE BARRAS PARA ANALIZAR CUALES SON LAS VARIABLES MAS IMPORTANTES 
coeficientes = model.params.drop('Intercept')
nombres_caracteristicas = coeficientes.index
nombres_caracteristicas
#ajuto el grafico
plt.figure(figsize=(10, 6))
plt.barh(nombres_caracteristicas, [coef for coef in coeficientes], color='skyblue')
plt.xlabel('Coeficientes')
plt.title('Caracteristicas mas importantes')
plt.gca().invert_yaxis() #esto lo hago para ver la mas significativa primero
plt.show()



#CALCULO DE RESIDUOS
residuos = model.predict(df[features]) - df["review_score"]
residuos

residuos.mean()

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
y_real = df['review_score']
X = df[features]
X = sm.add_constant(X)  
X = X.fillna(X.mean())
y_pred = model.predict(X)

# calculo del error cuadrado medio (MSE)
mse = mean_squared_error(y_real, y_pred)

#calculo el RMSE como la raíz cuadrada del MSE
rmse = np.sqrt(mse)
#imprimo el resultado
print("RMSE:", rmse)




#PLOTEAR LOS RESULTADOS
sns.histplot(residuos, kde=True, color='blue', bins=15)

#configuracion del grafico
plt.xlabel('Residuos')
plt.ylabel('Densidad')
plt.title('Histograma de Residuos')

plt.show()


#PLOTEAR LO DEL MODELO VS LO REAL
y_pred = model.predict(df[features])
y_real = df["review_score"]

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
y_pred = y_pred.dropna()
# KDE de datos reales
kde_real = gaussian_kde(y_real)
x_real = np.linspace(min(y_real), max(y_real), 100)
# Graficar la línea azul sin relleno y sin otros elementos
plt.plot(x_real, kde_real(x_real), color='blue', linestyle='-', fillstyle='none', label='review_score')

# KDE de datos predichos
kde_pred = gaussian_kde(y_pred)
x_pred = np.linspace(min(y_pred), max(y_pred), 100)
# Graficar la línea verde sin relleno y sin otros elementos
plt.plot(x_pred, kde_pred(x_pred), color='green', linestyle='-', fillstyle='none', label='predicted_review_score')

# Configura el gráfico
plt.xlabel('Puntuación de review')
plt.ylabel('Densidad')
plt.title('Distribución de review_score vs. predicted_review_score')
plt.legend()

# Muestra el gráfico
plt.grid(True)
plt.show()


#Usando seaborn, ya podríamos haber trazado una línea de regresión de review_score frente a real_vs_esperado
#Hacelo con una sub-muestra de la población, y una vez con la toda la población.
suborders = df.sample(10000) #agarro una muestra de mi poblacion, elegi que sea de 10000
sns.lmplot(x="real_vs_esperado", y="review_score", data=suborders) #uso como data la muestra
plt.title("Suborders: Review Score vs. Real_vs_esperado")
plt.ylim(0, None)
plt.show()

#ANALIZAR OUTLIERS
cinco_minimos_indices = df['columna de la que quiero ver outliers'].nsmallest(5).index
cinco_minimos_filas = df.loc[cinco_minimos_indices]
cinco_minimos_filas


#MODELO CON VARIABLES CATEGORICAS
features = ['C(columna)', 'C(columna)','C(columna)']
formula = "quiero predecir~ " + ' + '.join(features)
formula
modelcategoricas = smf.ols(formula,data=df)
modelcategoricas=modelcategoricas.fit()
modelcategoricas.summary()


#COLUMNA DE PREDICHO VS REAL
df['Predict'] = model.predict(df)
df_real_predict= df[['Net_Profit','Predict']]
df_real_predict['error']= df['Predict'] -df_real_predict['Net_Profit']
df_real_predict['error_cuad']= df_real_predict['error']**2

#GRAFICO DE RESIDUOS
sns.histplot(df_real_predict['error'], kde=True, edgecolor='w');

#error medio
df_real_predict['error'].mean()
format(df_real_predict['error'].mean(), '.15f') #para que no lo muestre en notacion cientifica

#error cuad medio
df_real_predict['error_cuad'].mean()



#GRAFICO DEL BENEFICIO NETO PARA CADA SECTOR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Filtrar y agrupar los datos de beneficio neto por sector
data = [df[df['Industry_Sector'] == sector]['Net_Profit'] for sector in df['Industry_Sector'].unique()]

# Elegir una paleta de colores de seaborn
sns.set_palette("Set3")

# Crear el gráfico de boxplot con colores de la paleta
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(data, labels=df['Industry_Sector'].unique(), patch_artist=True)

# Asignar colores de la paleta a los boxplots
for box in boxplot['boxes']: # Color de fondo
    box.set_edgecolor("gray")  # Color del borde

plt.xlabel('Sector')
plt.ylabel('Beneficio Neto')
plt.title('Boxplot del Beneficio Neto por Sector')
plt.tight_layout()
plt.show()


#CUANDO TENGO UNA COLUMNA CON VALORES COMO PALABRAS Y LE QUIERO ASIGNAR UN NUMERO A CADA UNO PARA DESPUES VER LA CORRELACION, LO QUE HACE ES SEPARARMELOS EN COLUMNAS Y PONER 1 SI ES TRUE Y 0 SI ES FALSE
from sklearn.preprocessing import OneHotEncoder

df_ohe = OneHotEncoder(sparse=False)
df_ohe.fit(df[['columna a encodear']])
df_ohe.categories_ 

df.columns
#copio los nombres de cada columna en los espacios de abajo

df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''],df[''] = df_ohe.fit_transform(df[['columna a encodear']]).T 

df.head()

#borro la columna que encodee 
df2=df.drop(columns=['columna que encodee'])
df2
plt.figure(figsize=(20,15))
sns.heatmap(df2.corr(),cmap = 'coolwarm',annot = True,annot_kws= {"size":8}) #hago el heatmap sobre las variables que encodee y las anteriores 


#grafico cada elemnto de la columna que encodee para cada columna original, entonces puedo ver como cada cultivo varia segun la temperatura,potasio,humedad,etc
sns.set(style='whitegrid')
columns_to_plot= ['','','','','','','']
fig,axes = plt.subplots(len(columns_to_plot),1,figsize=(12,5*len(columns_to_plot)))

for i,column in enumerate(columns_to_plot):
    sns.boxplot(data=df, x='columna que encodee',y=column, ax=axes[i])
    axes[i].set_ylabel(f'{column}')
    axes[i].set_xlabel('columna que encodee')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout() #para que no se pisen los graficos
plt.show()



#ARBOL DE DECISION
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['columna'] = LE.fit_transform(df['columna'])

df.sample(5)



from sklearn.model_selection import train_test_split
X = df[['','', '', '' ,'', '', '']] #nombres de las columnas
y = df[['columna']] #nombre de la columna que quiero ver como varia de acuerdo a las otras

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#DESDEACA 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
# Crear el modelo
decision_tree_model = DecisionTreeClassifier(criterion="entropy", random_state=2,
max_depth=5) #max_depth son los niveles que va a tener mi arbol, y random_state es mi semilla 

decision_tree_model.fit(X_train, y_train)

# Validación cruzada
score = cross_val_score(decision_tree_model, X, y, cv=5)  #va a separar en 5 diferentes #me muestra la prepcision del modelo
print('Puntuación de validación cruzada:', score)
# Precisión en entrenamiento
dt_train_accuracy = decision_tree_model.score(X_train, y_train)
print("Precisión en entrenamiento =", dt_train_accuracy)
# Precisión en pruebas
dt_test_accuracy = decision_tree_model.score(X_test, y_test)
print("Precisión en pruebas =", dt_test_accuracy)
#HASTA ACA


#MATRIZ DE CONFUSION
y_pred = decision_tree_model.predict(X_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_true, y_pred)
# Visualización de la matriz de confusión
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(cm_dt, annot=True, linewidth=0.5, fmt=".0f", cmap='viridis', ax=ax)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title('Matriz de Confusión')
plt.show()

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn import tree


#LO DE ACA ABAJO ME ARMA EL ARBOL (TENGO QUE HACER LO DE ARRIBA PRIMERO)
labels= X.columns
targets = LE.classes_
data = export_graphviz(decision_tree_model, out_file=None, feature_names=labels, class_names=targets, filled=True, rounded=True, special_characters=True)
graph= graphviz.Source(data)
graph   


#REGRESION LOGISTICA
import statsmodels.api as sm
logist_model=sm.MNLogit(y_train,sm.add_constant(X_train))
logist_model
result=logist_model.fit(method="bfgs")
stats1=result.summary()
print(stats1)




from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as  np
y_pred = result.predict(sm.add_constant(X_test))
y_pred_labels = np.argmax(y_pred.values, axis=1)

accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred_labels))
cm = confusion_matrix(y_test, y_pred_labels)
print('Confusion Matrix:')
# print(cm)

# Visualización de la matriz de confusión
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(cm, annot=True, linewidth=0.5, fmt=".0f", cmap='viridis', ax=ax)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title('Matriz de Confusión')
plt.show()



#TENGO QUE COMPARAR LA ACURRACY DEL DE REGRESION Y DEL ARBOL.
'''
