import pickle
import pandas as pd
import tensorflow as tf
import json

with open('/Users/leonnayd/Desktop/forecastCostMedicalInsurance/models/linear/model.pickle', mode='rb') as filestream:
    linear_model = pickle.load(filestream)

with open('/Users/leonnayd/Desktop/forecastCostMedicalInsurance/models/linear/info.json', 'r', encoding='utf-8') as filestream:
    info = json.load(filestream)

def calculate(age: float) -> float:
    ans = linear_model.predict([[age]])[0][0]
    return ans

def get_r2() -> float:
    return info['r2']

def get_rmse() -> float:
    return info['rmse']