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


class InputValue:
    def __init__(self, value):
        self.value = value
        self.predictor = None
        self.predicted_value = None

    def predict(self):
        assert not self.predictor
        self.predicted_value = self.predictor.predict(self.value)