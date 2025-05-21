import pickle
import numpy as np


# load the model from disk
# fileName_xgb = '/Users/yanjiasun/Documents/Yanjia_code/WQI-Delaware-ML-WS/Model/xgboost.pkl'
fileName_xgb = 'xgboost.pkl'

loaded_model = pickle.load(open(fileName_xgb, 'rb'))
print('Loaded ' +fileName_xgb+ ' successfully!!!')

def predict_water_quality(features):
    # features: list or np.array
    prediction = loaded_model.predict(features)
    # prediction = loaded_model.predict(np.array([features]))

    return prediction[0]