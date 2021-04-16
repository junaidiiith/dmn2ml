import csv
import itertools

import pandas as pd
import numpy as np
import math
import xgboost as xgb
import pickle
import csv

from sklearn import preprocessing
from matplotlib import pyplot as plt
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from keras.models import Sequential
from keras import backend
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from catboost import CatBoostRegressor, cv

SEED = 123
XGB_EPOCH_NR = 2500
np.random.seed(SEED)
RESULTS = list()


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def MAPE(y_test, y_pred):
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mape


def writeToFile(dataset, header, filename):
    try:
        with open(filename, 'w', newline='') as fl:
            w = csv.writer(fl)
            w.writerow(header)
            for row in dataset:
                w.writerow(row)
    except IOError:
        print("I/O Error occurred here!!")


def denormalize(X):
    return np.array([float("{:.2f}".format(i)) for i in map(lambda x: (maxVal - minVal) * x + minVal, X)])


def evalModels(X_test, y_test, models):
    results_dict = {}
    percentage_trends = dict()
    print("Number of unique values in training set are: " + str(len(np.unique(y_test))) + "\n")

    for model, modelNameColor in models:
        modelName, color = modelNameColor
        if modelName == "xgb":
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)
        elif modelName == "neuralNet":
            y_pred = model.predict(X_test)[:, 0]
        else:
            y_pred = model.predict(X_test)
        print("\n\nEvaluating " + modelName)
        print("Number of unique values in prediction are: " + str(len(np.unique(y_pred))) + "\n")
        rms = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = MAPE(y_test, y_pred)
        results_dict[modelName] = {"rmse": rms, "mae": mae, "mape": mape}
        percentage_trends[modelNameColor] = get_percentage_trend(y_test, y_pred)
        print("\nDone evaluating: " + modelName)

    RESULTS.append(percentage_trends)
    # plot_percentage_trend(percentage_trends)
    return results_dict


def get_percentage_trend(y_test, y_pred, MAX_PERCENTAGE_ERROR=0.1):
    errors = list()
    i = 0.0
    while i < MAX_PERCENTAGE_ERROR:
        # p = float("{:.6f}".format(i))
        actual_y_test, actual_y_pred = denormalize(y_test), denormalize(y_pred)
        diff = abs(actual_y_test - actual_y_pred)
        correctly_predicted_mean_sq = len(diff[diff < i * actual_y_test])
        errors.append(
            (float("{:.2f}".format(i * 100)), float("{:.2f}".format(correctly_predicted_mean_sq / len(y_test)))))

        i += 0.0001

    return errors


fire_hazard_class_factor = [0.00026, 0.00051, 0.00073, 0.00116, 0.00145, 0.00189, 0.00305, 0.00585]
contribution_margin = [i for i in range(10000, 99000000, 1000000)]
liability_time = [6, 12, 18]
benefit_cover = [0, 1]
benefit_gastro = [0, 1]
benefit_trade = [0, 1]
elementary_liability = [3, 6, 9]
special_discount = [i for i in range(-100, 30, 10)]
vpc_discount = [i for i in range(1, 100, 10)]

fire_bonus_header = ["Contribution Margin", "Liability Time", "Benefit Trade", "Benefit Gastro", "Fire Bonus"]
elementary_bonus_header = ["Contribution Margin", "Elementary Liability", "Elementary Bonus"]
special_discount_header = fire_bonus_header[:-1] + ["Elementary Liability"] + \
                          ["Special Discount", "VPC Discount", "Insurance"]

elementary_bonus_dataset = list()
elementary_bonus_values_map = dict()
elementary_bonus_values = list()
for element in itertools.product(contribution_margin, elementary_liability):
    if element[1] == 3:
        X = element[0] * 0.00044 * 1.15
    elif element[1] == 6:
        X = element[0] * 0.00058 * 1.15
    elif element[1] == 9:
        X = element[0] * 0.00073 * 1.15

    X = float("{:.3f}".format(X))
    elementary_bonus_dataset.append((element[0], element[1], X))
    elementary_bonus_values_map[X] = (element[0], element[1])
    elementary_bonus_values.append(X)


fire_bonus_dataset = list()
fire_bonus_values = dict()
for element in itertools.product(contribution_margin, liability_time, benefit_trade, benefit_gastro,
                                 fire_hazard_class_factor):
    if element[1] == 6:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * 0.75 * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 0.75 * 1.15

    elif element[1] == 12:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 1.15
    elif element[1] == 18:
        if element[2]:
            if element[3]:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06 + 0.06)
            else:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06)
        else:
            if element[3]:
                X = element[0] * element[4] * 1.25 * (1.15 + 0.06)
            else:
                X = element[0] * element[4] * 1.25 * 1.15
    X = float("{:.3f}".format(X))
    fire_bonus_dataset.append((element[0], element[1], element[2], element[3], X))
    fire_bonus_values.append(X)

# print(len(elementary_bonus_dataset))
# print(len(fire_bonus_dataset))
writeToFile(fire_bonus_dataset, fire_bonus_header, "firebonus_dataset_old.csv")
writeToFile(elementary_bonus_dataset, elementary_bonus_header, "elementary_dataset_old.csv")

try:
    with open("special_discount_old.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(special_discount_header)
        for element in itertools.product(special_discount, vpc_discount, elementary_bonus_values, fire_bonus_values):
            X = (element[2] + element[3]) * ((100 - element[0]) * (100 - element[1]) / 100)
            X = float("{:.2f}".format(X))
            writer.writerow((element[0], element[1], X))
except IOError:
    print("I/O Error occurred here!!")


