import pandas as pd
import numpy as np

"""
Este modulo contiene todas las funciones de utilidad a la hora
de calcular las probabilidades y frecuencias del modelo bayesiano
"""

def calculate_relative_frequency(dataset, feature, targetClass):
    """
    Calcula la probabilidad, en cada una de las clases de salida, segun
    la caracteristica a analizar (los nc)
    """
    frequencies = calculate_absolute_frequency(dataset, targetClass)
    ncs = calculate_nc(dataset, feature, targetClass)
    result = ncs.copy()
    uniqueValues = pd.unique(dataset[feature])
    for category in ncs.index.categories:
        for column in ncs.columns.categories:
            nc = ncs[column][category]
            m = len(dataset.columns) # m es una constante, la cantidad de features
            p = 1/len(uniqueValues) # p es la probabilidad a priori, 1 divido entre la cantidad de clases
            n = frequencies[category] # n es la cantidad de observaciones donde este presente cada una de las categorías
            result.loc[category, column] = bayes_frequency(nc, m, p, n)

    return ncs, result

def calculate_relative_frequency_range(dataset, feature, targetClass, start, end, step):
    """
    Es la contraparte de la funcion 'calculate_relative_frequency', pero para
    valores cuyos atributos son rangos de datos
    """
    frequencies = calculate_absolute_frequency(dataset, targetClass)
    ncs = calculate_nc_range(dataset, feature, targetClass, start, end, step)
    result = ncs.copy()
    uniqueValues = pd.unique(dataset[feature])
    for category in ncs.index.categories:
        for column in np.asarray(ncs.columns):
            nc = ncs[column][category]
            m = len(dataset.columns)
            p = 1/len(uniqueValues)
            n = frequencies[category]
            result.loc[category, column] = bayes_frequency(nc, m, p, n)

    return ncs, result

"""
    Cuenta la cantidad de veces que se observa cada una de las clases, es decir,
    su frecuencia absoluta
"""
def calculate_absolute_frequency(dataset, targetClass):
    selectedColumns = dataset[[targetClass]]
    selectedColumns = selectedColumns.astype({targetClass: "category"})
    return selectedColumns.groupby(targetClass).size()

"""
    Calcula los NC del modelo, las observaciones agrupadas por ambas la categoría
    y la clase de interes por analizar
"""
def calculate_nc(dataset, feature, targetClass):
    selectedColumns = dataset[[feature, targetClass]]
    selectedColumns = selectedColumns.astype({targetClass: "category"})
    selectedColumns = selectedColumns.astype({feature: "category"})
    grouped = selectedColumns.groupby([feature, targetClass])
    return grouped.size().unstack(level=0)

"""
    Es la contraparte de la funcion 'calculate_nc', pero para
valores cuyos atributos son rangos de datos
"""
def calculate_nc_range(dataset, feature, targetClass, start, end, step):
    selectedColumns = dataset[[feature, targetClass]]

    selectedColumns = selectedColumns.astype({targetClass: "category"})

    grouped = selectedColumns.groupby([feature, targetClass], dropna=False)

    result = grouped.size().unstack(level=0)
    number_of_classes = len(result.index.categories)

    """ 
    Si el valor no se encuentra en el dataset, agregarlo para asi abarcar todas las posibilidades
    definidas en el rango
    """

    for i in np.arange(start, end, step):
        if i not in result:
            new_column = pd.Series(np.zeros((number_of_classes)).transpose())
            result.insert(0, i, new_column)
            

    result = result.fillna(0)
    return result

    


def make_prediction(relative_frequencies):
    total_freq = relative_frequencies.prod(axis=1)
    print(total_freq)
    return np.asarray(total_freq.index)[total_freq.argmax()]

def bayes_frequency(nc, m, p, n):
    return (nc+m*p)/(n+m)