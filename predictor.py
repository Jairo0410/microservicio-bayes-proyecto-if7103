from config import constants
import repository
from utils import make_prediction
import pandas as pd
import bayes

def guessPersonType(sex, headquarter, grade):
    model = bayes.BayesModel('Estudiantes', constants.labels.estilos.style)
    model.add_feature(constants.labels.estilos.sex)
    model.add_feature(constants.labels.estilos.headquarter)
    model.add_numerical_feature(constants.labels.estilos.grade, 0, 10, 0.1)

    model.loadModel()

    sex_probs = model.getProbabilities(constants.labels.estilos.sex, sex)
    headquarter_probs = model.getProbabilities(constants.labels.estilos.headquarter, headquarter)
    grade_probs = model.getProbabilities(constants.labels.estilos.grade, grade)

    return make_prediction(pd.concat([sex_probs, headquarter_probs, grade_probs], axis=1))
    