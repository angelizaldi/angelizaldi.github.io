---
summary: Aprende a realizar un cross validation con la librería sklearn.
authors:
  - admin
lastMod: 2023-01-25T00:00:00.000Z
title: Cross Validation con Scikit-Learn en Python
subtitle: Aprende a realizar un cross validation con la librería sklearn.
date: 2023-01-25T00:00:00.000Z
tags:
  - machine learning
  - python
  - sklearn
categories:
  - machine learning
  - python
  - sklearn
projects: []
image:
  caption: ""
  focal_point: LEFT
  filename: featured.png
---

En este tutorial revisaremos como realizar un cross validation usando la librería `scikit-learn`. Para este tutorial se da por hecho que el lector está familiarizado con conceptos básicos de machine learning, con los modelos de decision tree y con la API de la librería `scikit-learn`. 

## Cross Validation

Cross validation es una técnica en la que en un proceso iterativo el train set se divide en _folds_. 

Existen diversas técnicas para hacer la división del training set en folds. En la imagen se muestra la técnica de "_k-fold"_. En esta técnica el train set se divide en folds consecutivos del mismo tamaño, el número de folds es igual al número de iteraciones. En cada iteración un fold se utiliza como validation set y el resto como train set.

Dividir el train set de esta forma y entrenar diversos modelos permite estimar la variabilidad del desempeño del modelo.

Otras técnicas para definir los folds se pueden encontrar en la [documentación](https://scikit-learn.org/stable/modules/classes.html#splitter-classes) de `scikit-learn`. En este tutorial haremos uso de _k-folds_.

## Importación de datos

Antes de aplicar el cross validation primero importaremos los datos y definiremos un modelo sencillo. El dataset que utilizaremos para este tutorial es el `California Housing` dataset el cual podemos importar del módulo `datasets` de `scikit-learn`


```python
# Importar dataset
from sklearn.datasets import fetch_california_housing

# Recuperar la feature matrix y el target vector
housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target
```

Para obtener más información de este dataset visitar [esta pagína](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

El objeto `data` es un `pd.DataFrame`, mientras que `target` es un `pd.Series`.

Revisemos las primeras filas con el método `DataFrame.head()`.


```python
# Imprimir las primeras 5 filas
display(data.head())
```

|   | MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Latitude | Longitude |
|---|--------|----------|----------|-----------|------------|----------|----------|-----------|
| 0 | 8.33   | 41.0     | 6.98     | 1.02      | 322.0      | 2.56     | 37.88    | -122.23   |
| 1 | 8.3    | 21.0     | 6.24     | 0.97      | 2401.0     | 2.11     | 37.86    | -122.22   |
| 2 | 7.26   | 52.0     | 8.29     | 1.07      | 496.0      | 2.8      | 37.85    | -122.24   |
| 3 | 5.64   | 52.0     | 5.82     | 1.07      | 558.0      | 2.55     | 37.85    | -122.25   |
| 4 | 3.85   | 52.0     | 6.28     | 1.08      | 565.0      | 2.18     | 37.85    | -122.25   |


Ahora revisemos la información del dataframe


```python
# Imprimir la información de data
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   MedInc      20640 non-null  float64
     1   HouseAge    20640 non-null  float64
     2   AveRooms    20640 non-null  float64
     3   AveBedrms   20640 non-null  float64
     4   Population  20640 non-null  float64
     5   AveOccup    20640 non-null  float64
     6   Latitude    20640 non-null  float64
     7   Longitude   20640 non-null  float64
    dtypes: float64(8)
    memory usage: 1.3 MB
    None


Podemos observar que `data` no tiene valores perdidos y que todas las columnas son de tipo numérico. 

Ahora revisemos la información de `target`.


```python
# Imprimir la información de target
print(target.describe())
```

    count    20640.000000
    mean         2.068558
    std          1.153956
    min          0.149990
    25%          1.196000
    50%          1.797000
    75%          2.647250
    max          5.000010
    Name: MedHouseVal, dtype: float64


Podemos observar que `target` es una columna numérica entre el rango 0.15 y 5. Esta variable representa el valor medio de las casas por distrito en ciento de miles de dolares, es decir $100,000.00.

## Definir el modelo

Antes de definir el modelo primero haremos un train-test split con la función `train_test_split` del módulo `sklearn.model_selection`.


```python
# Importar la función
from sklearn.model_selection import train_test_split

# Crear el train-test split
data_train, data_test, target_train, target_test = train_test_split(data, target)
```

Ahora definamos un modelo sencillo, para ello utilizaremos un `DecisionRegressor` del módulo `sklearn.tree`. 


```python
# Importar el modelo
from sklearn.tree import DecisionTreeRegressor

# Definir el modelo
tree_regressor = DecisionTreeRegressor()
```

Ajustaremos el modelo al train set.


```python
tree_regressor.fit(data_train, target_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor()</pre></div></div></div></div></div>



Con el modelo entrenado procederemos a evaluar el test set con el método `DecisionRegressor.score()`.


```python
score = tree_regressor.score(data_test, target_test)
print(f"Score: {score:.2f}")
```

    Score: 0.61


## Aplicar un cross validation

Para tener un panorama más amplio de qué tan bueno es el modelo procedemos a aplicar un cross validation con una estrategia de _k-folds_, para ello usaremos la clase `KFold` de `sklearn.model_selection`. Se debe definir el número de folds en el parámetro `n_splits`, para este turial usaremos 10 folds.

> Tener en cuenta que _k_fold_ puede ser costoso computacionalmente, por ello es recomendado para datasets no muy grandes o no utilizar mucho `n_splits`.


```python
# Importar la clase
from sklearn.model_selection import KFold

# Crear la estrategia de splitting
cv = KFold(n_splits=10)
```

Una vez definida la estrategia de splittling procedemos a calcular los scores en cada iteración con la función `cross_val_score` del módulo `sklearn.model_selection`. Esta función evalua un _score_ para cada iteración, en este caso, como usamos `n_splits=10` entonces se entrenarán 10 modelos diferentes y por lo tanto tendremos 10 scores. 

Los principales parámetros de esta función son:
- `estimator`: Es el modelo a entrenar, en nuestro caso es el objeto `tree_regressor`.
- `X`: Datos correspondientes a la feature matrix.
- `y`: Datos correspodientes al target vector.
- `cv`: Estrategia para el splitting, en este caso corresponde al objeto `cv` definido anteriormente.
- `scoring`: Métrica a usar, en nuestro caso usaremos $R^2$, para conocer más sobre esta métrica se puede visitar su [artículo en Wikipedia](https://es.wikipedia.org/wiki/Coeficiente_de_determinaci%C3%B3n). Para revisar otras métricas existente visitar la [documentación](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values) de `sklearn`.

> **Nota**: Si en `cv` se indica un número entero `n`, entonces por default se calculará un _k-fold_ con `n` folds. En este tutorial se usó la clase `KFold` para ser más explícitos, pero el resultado hubiera sido el mismo si simplemente se hubiera usado `cv=10`. 

Los datos que se deben de usar en un cross validation son los correspondientes al _train_set_, es decir, a `data_train` y `target_train`.


```python
# Importar la función
from sklearn.model_selection import cross_val_score

# Calcular los score
scores = cross_val_score(estimator=tree_regressor,
                        X=data_train,
                        y=target_train,
                        cv=cv,
                        scoring="r2")
```

Con los scores calculados podemos imprimirlos para visualizarlos, el objeto retornado es un `ndarray`.


```python
# Imprimir los scores
print(scores)
```

    [0.55019071 0.54489269 0.53451175 0.63878763 0.6261445  0.59789409
     0.59864349 0.61077492 0.58288147 0.62066303]


Podemos ver que los scores son todos diferentes. Para estimar cuál será el desempeño del modelo podemos calcular el promedio y la desviación estándar. 


```python
print(f"r2 de un DecisionTreeRegressor: {scores.mean():.2f} +/- {scores.std():.2f}")
```

    r2 de un DecisionTreeRegressor: 0.59 +/- 0.03


Ahora podemos estar más seguros cuál será el desempeño del modelo ante datos nuevos. 

Para finalizar es importante resaltar que alternativamente se pudo haber usado la función `cross_validate()` del módulo `sklearn.model_selection`. Esta función no solo retorna los scores en el validation set, sino que también puede retornar:
- `test_score`: Scores del test set, que correponde al validation set.
- `train_score`: Scores del train set.
- `fit_time`: Tiempo que tardó en entrenarse cada modelo.
- `score_time`: Tiempo que tardó en calcularse cada score.
- `estimator`: Modelos entrenados. Estos modelos se pueden utilizar para evaluar nuevos datos y se puede utilizar el `test_score` para elegir el modelo con el mejor desempeño.
