from config import constants
import repository
from utils import make_prediction
import pandas as pd
import bayes

def guessPersonType(clima, ambiente, agua, zona, distancia):
    model = bayes.BayesModel('Atractivos', constants.labels.personas.tipoPersona)
    model.add_feature(constants.labels.personas.clima)
    model.add_feature(constants.labels.personas.distancia)
    model.add_feature(constants.labels.personas.zona)
    model.add_feature(constants.labels.personas.ambiente)
    model.add_feature(constants.labels.personas.agua)

    model.loadModel()

    return make_prediction(pd.concat([
        model.getProbabilities(constants.labels.personas.clima, clima),
        model.getProbabilities(constants.labels.personas.distancia, distancia),
        model.getProbabilities(constants.labels.personas.zona, zona),
        model.getProbabilities(constants.labels.personas.ambiente, ambiente),
        model.getProbabilities(constants.labels.personas.agua, agua)
    ], axis=1))
    