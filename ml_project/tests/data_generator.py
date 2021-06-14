"""Distribution parameters were taken from EDA"""
import pandas as pd
import numpy as np


def generate_negative():
    age = np.random.randint(53, 61)
    sex = np.random.choice([0, 1], p=[0.2, 0.8])
    cp = np.random.randint(0, 2)
    trestbps = np.random.normal(134, 18)
    chol = np.random.normal(251, 49)
    fbs = 0
    restecg = 0
    thalach = np.random.normal(139, 22)
    exang = 1
    oldpeak = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
    slope = np.random.choice([0, 1, 2], p=[0.1, 0.7, 0.2])
    ca = 1
    thal = np.random.choice([0, 1, 2, 3], p=[0.0, 0.0, 0.5, 0.5])
    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': 0
    }


def generate_positive():
    age = np.random.randint(49, 56)
    sex = np.random.choice([0, 1], p=[0.2, 0.8])
    cp = np.random.randint(1, 3)
    trestbps = np.random.normal(129, 16)
    chol = np.random.normal(242, 53)
    fbs = 0
    restecg = 1
    thalach = np.random.normal(158, 19)
    exang = 0
    oldpeak = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
    slope = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
    ca = 0
    thal = np.random.choice([0, 1, 2, 3], p=[0.0, 0.2, 0.5, 0.3])
    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': 1
    }


def generate_dataset(num_each_class=100):
    data = [generate_positive() for _ in range(num_each_class)] +\
           [generate_negative() for _ in range(num_each_class)]
    df = pd.DataFrame(data)
    return df
