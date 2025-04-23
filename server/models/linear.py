import pickle
import json
import os

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)

# Construct paths relative to the script location
model_path = os.path.join(current_dir, '..', '..', 'models', 'linear', 'model.pickle')
info_path = os.path.join(current_dir, '..', '..', 'models', 'linear', 'info.json')

# Load the model and info
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(info_path, 'r', encoding='utf-8') as f:
    info = json.load(f)

def calculate(age: float) -> float:
    """Predict insurance cost based on age"""
    ans = model.predict([[age]])[0][0]
    return ans

def get_r2() -> float:
    """Get R-squared score"""
    return info['r2']

def get_rmse() -> float:
    """Get RMSE score"""
    return info['rmse']