import utils
import repository
import bayes
from config import constants

atractivos_data = repository.getAtractivosData()

# ------- Modelo para predecir estilo en los estudiantes (Variables de Kolb) -------------
atractivosModel = bayes.BayesModel('Atractivos', constants.labels.personas.tipoPersona)
atractivosModel.add_feature(constants.labels.personas.clima)
atractivosModel.add_feature(constants.labels.personas.distancia)
atractivosModel.add_feature(constants.labels.personas.zona)
atractivosModel.add_feature(constants.labels.personas.ambiente)
atractivosModel.add_feature(constants.labels.personas.agua)

atractivosModel.fitModel(atractivos_data)
atractivosModel.saveModel()