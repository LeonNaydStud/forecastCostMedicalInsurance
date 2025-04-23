import pickle
import pandas as pd
import tensorflow as tf
import json

model = tf.keras.models.load_model('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/model.keras')

with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/info.json', 'r', encoding='utf-8') as filestream:
    info = json.load(filestream)

with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/normalizer.pickle', mode='rb') as filestream:
    scaler = pickle.load(filestream)

def def_sex(sex_str: str) -> int:
    sex_dict = {
        'Man': 1,
        'Woman': 0
    }
    return sex_dict[sex_str]

def def_children(children_str: bool) -> int:
    children_dict = {
        True: 1,
        False: 0
    }
    return children_dict[children_str]

def normalize(age: float, children: str, bmi: float, sex_male: str):
    dfXNorm = pd.DataFrame (data = scaler['x'].transform([[age, children, bmi, sex_male]]),
                            columns=['age','children','bmi','sex_male'])

    return dfXNorm

def calculate(age: float, children: str, bmi: float, sex_male: str) -> float:
    child = def_children(children)
    sex = def_sex(sex_male)
    dfXNorm = normalize(age, child, bmi, sex)
    print(dfXNorm)
    with tf.device('/CPU:0'):
        yNorm_pred = model.predict( dfXNorm [['age', 'children', 'bmi', 'sex_male',]])
    ans = scaler['y'].inverse_transform(yNorm_pred)[0][0]
    return ans, dfXNorm, yNorm_pred

def get_r2() -> float:
    return info['r2']

def get_rmse() -> float:
    return info['rmse']