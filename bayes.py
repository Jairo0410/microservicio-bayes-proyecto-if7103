import utils
import repository
import pandas as pd

class BayesModel:
    """
    La siguiente clase permite abstraer el procedimiento de almacenamiento de nombres
    de atributos, la variable a predecir y guardar las probabilidades para hacer predicciones
    """
    def __init__(self, modelName, modelClass):
        self.features = []
        self.modelClass = modelClass
        self.modelName = modelName
        self.model = {}
        self.classCategories = []

    
    def add_feature(self, featureName):
        """
        Agrega una caracteristica categorica al objeto que representa el modelo
        """
        newFeature = {
            "name": featureName,
            "isRange" : False
        }

        self.features.append(newFeature)

    def add_numerical_feature(self, featureName, start=0, end=0, step=1):
        """
        Agrega una caracteristica numerica al objeto que representa el modelo,
        junto con su límite superior e inferior
        """

        newFeature = {
            "name": featureName,
            "isRange": True,
            "start": start,
            "end" : end,
            "step" : step
        }

        self.features.append(newFeature)

    def fitModel(self, dataset):
        """
        Calcula las probabilidades de cada uno de los atributos almacenados en
        este objeto, en caso de ser una variable señalada como de rango, llama a una
        funcion especializada para encontrar su probabilidad
        """
        for feature in self.features:
            featureName = feature["name"]
            isRange = feature["isRange"]

            if isRange :
                start = feature["start"]
                end = feature["end"]
                step = feature["step"]
                ncs, probabilities = utils.calculate_relative_frequency_range(dataset, featureName, self.modelClass, start, end, step)
            else :
                ncs, probabilities = utils.calculate_relative_frequency(dataset, featureName, self.modelClass)
            self.model[featureName] = {
                "frequencies": ncs,
                "probabilities": probabilities
            }
    
    def saveModel(self):
        """
        Almacena las probabilidades del modelo y las frecuencias absolutas,
        para su persistencia y posterior consulta
        """
        for feature in self.features:
            featureName = feature["name"]
            modelProbs = self.model[featureName]["probabilities"]
            modelFreqs = self.model[featureName]["frequencies"]
            repository.saveProbabilites(modelProbs, self.modelName, featureName, self.modelClass)
            repository.saveFrequences(modelFreqs, self.modelName, featureName, self.modelClass)

    def loadModel(self):
        """
        Carga los contenidos de los archivos donde se encuentra guardado el modelo,
        dadas las caracteristicas o atributos del modelo
        """
        for feature in self.features:
            featureName = feature["name"]
            probabilities = repository.readProbabilities(self.modelName, featureName, self.modelClass)
            probabilities = probabilities.set_index(self.modelClass)

            modelForFeature = {
                "probabilities": probabilities
            }
            self.model[featureName] = modelForFeature

    def getModel(self):
        return self.model

    def getProbabilities(self, featureName, value):
        """
        Retorna las probabilidades para cada una de las clases, dado un valor
        exacto en una caracteristica o feature, ejemplo:

        ACOMODADOR: 0.001
        ASIMILADOR: 0.011
        CONVERGENTE: 0.025
        DIVERGENTE: 0.02
        """
        probs = self.model[featureName]["probabilities"]
        return probs[value]
            
