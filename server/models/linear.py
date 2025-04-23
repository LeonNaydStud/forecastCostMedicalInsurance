import pickle
from sklearn import linear_model
import json

with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/model.pickle', mode='rb') as filestream:
    linear_model = pickle.load(filestream)

with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/info.json', 'r', encoding='utf-8') as filestream:
    info = json.load(filestream)

def calculate(age: float) -> float:
    ans = linear_model.predict([[age]])[0][0]
    return ans

def get_r2() -> float:
    return info['r2']

def get_rmse() -> float:
    return info['rmse']