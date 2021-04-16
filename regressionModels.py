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


def plot_percentage_trend(d):
    plt.xlabel("Percentage Error")
    plt.ylabel("Accuracy")
    for modelNameColor, points in d.items():
        X = [element[0] for element in points]
        y = [element[1] for element in points]
        plt.plot(X, y, label=modelNameColor[0], color=modelNameColor[1])
    plt.legend()
    plt.show()


def denormalize(X):
    return np.array([float("{:.2f}".format(i)) for i in map(lambda x: (maxVal - minVal) * x + minVal, X)])


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


def pre_process_data(raw_data, preprocessing_headers):
    le = preprocessing.LabelEncoder()
    mmc = preprocessing.MinMaxScaler()
    raw_data[preprocessing_headers['mmc']] = mmc.fit_transform(raw_data[preprocessing_headers['mmc']])
    if len(preprocessing_headers['le']) > 0:
        raw_data[preprocessing_headers['le']] = le.fit_transform(raw_data[preprocessing_headers['le']])
    return raw_data.dropna()


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

    # RESULTS.append(percentage_trends)
    plot_percentage_trend(percentage_trends)
    return results_dict


def fitModels(X_train, y_train, X_test, y_test, prefix=''):
    modelList = []

    model = Lasso()
    model.fit(X_train, y_train)
    modelList.append((model, ("Lasso", "#ff9933")))
    pickle.dump(model, open("models/" + prefix + "lasso.sav", 'wb'))

    model = Ridge()
    model.fit(X_train, y_train)
    modelList.append((model, ("Ridge", "#33ff33")))
    pickle.dump(model, open("models/" + prefix + "ridge.sav", 'wb'))

    # model = Sequential()
    # model.add(Dense(12, input_dim=len(X_train.columns), kernel_initializer='normal', activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # model.summary()
    # model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    # model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1, validation_split=0.2)
    # modelList.append((model, ("neuralNet", "#cc99ff")))
    # model_json = model.to_json()
    # with open("models/" + prefix + "nn.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("models/" + prefix + "nn.h5")

    model = CatBoostRegressor(iterations=400,
                              learning_rate=0.02,
                              depth=12,
                              eval_metric='RMSE',
                              random_seed=23,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=75,
                              od_wait=100)
    model.fit(X_train, y_train,
              eval_set=(X_test, y_test),
              use_best_model=True,
              verbose=True)
    pickle.dump(model, open("models/" + prefix + "cb.sav", 'wb'))
    modelList.append((model, ("CatBoost", "#0000ff")))

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    modelList.append((model, ("Regression tree", "#99ffff")))
    pickle.dump(model, open("models/" + prefix + "decisionTree.sav", 'wb'))

    tree.export_graphviz(model, out_file="figures/treeDotfile.dot",
                         feature_names=X_train.columns,
                         filled=True)

    #     fig = pyplot.figure(figsize=(30, 30))
    #     _ = tree.plot_tree(model, feature_names=X_train.columns, filled=True)
    #     pyplot.savefig("figures/decisionTree.png")
    #     pyplot.clf()
    # fitting the cgboost model requires a slightly different dataformat:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # define model hyperparamters
    param = {
        'max_depth': 15,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'nthread': 16,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        'eval_metric': 'rmse'
    }
    num_round = XGB_EPOCH_NR
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    # fitting the model

    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    bst.save_model("models/" + prefix + "xgb.model")

    modelList.append((bst, ("xgb", "#ff0000")))

    return modelList


def crossValidation(data, output_variable_name):
    """
	This function get the dataframe as input and performs 5 fold crossvalidation on the fitted models
	for the metric RMSE is used
	"""
    X, xt, y, yt = train_test_split(
        data.drop(output_variable_name, axis=1), data[output_variable_name], test_size=0.01, random_state=SEED)

    model = pickle.load(open("models/lasso.sav", 'rb'))
    lassoCV = -mean(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))

    model = pickle.load(open("models/ridge.sav", 'rb'))
    ridgeCV = -mean(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))

    model = pickle.load(open("models/decisionTree.sav", 'rb'))
    decTreeCV = -mean(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))

    param = {
        'max_depth': 15,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'nthread': 16,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        'eval_metric': 'rmse'
    }
    num_round = XGB_EPOCH_NR

    dtrain = xgb.DMatrix(X, label=y)
    xgbCV = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_round,
        seed=SEED,
        nfold=5,
        metrics={'rmse'}
    )["test-rmse-mean"][-1:]

    param = {
        "iterations": 400,
        "learning_rate": 0.02,
        "depth": 12,
        "eval_metric": 'RMSE',
        "random_seed": 23,
        "bagging_temperature": 0.2,
        "od_type": 'Iter',
        "metric_period": 75,
        "od_wait": 100
    }

    catBoostCV = cv(data, param, fold_count=5, plot=True)

    return lassoCV, ridgeCV, decTreeCV, xgbCV, catBoostCV


def get_outVar(file):
    with open(file) as fl:
        reader = csv.reader(fl)
        for i in reader:
            return i[-1].strip()


def train_regression_model_on_data(file_name, preprocessing_headers, prefix=''):
    data = pd.read_csv(file_name, sep=",")
    output_variable_name = get_outVar(file_name)
    print("Data Preprocessing...")
    data = pre_process_data(data, preprocessing_headers)
    print("Data Preprocessed")
    if len(data) < 100:
        print("Too less data points for ML. Please proceed with DMN approach only " + str(
            len(data)) + " data points" + "\n")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(output_variable_name, axis=1), data[output_variable_name], test_size=0.20, random_state=SEED)
    print("Start training models\n")
    modList = fitModels(X_train, y_train, X_test, y_test, prefix)

    print("Training models completed. Starting Evalation" + "\n")
    # evaluating the models

    resultDicts = evalModels(X_test, y_test, modList)
    for key, value in resultDicts.items():
        print(str(key) + str(value) + "\n")


# a, b, c, d = crossValidation(data, OUTPUT_VAR_NAME)

# print("5 fold mean rmse cv results"+"\n")
# stroing the CV results in a dictionary
# cvResultsDict = {"Lasso": {"rmse": a}, "Ridge": {"rmse": b},
# 				 "Regression Tree": {"rmse": c}, "XGBoost": {"rmse": d}}

# for key,value in cvResultsDict.items():
# 	print(str(key) + str(value)+"\n")


elementary_data_preprocessing = {
    'mmc': ['Garage Price', 'Garage Square Feet'],
    'le': ['Garage Condition', ]
}
firebonus_data_preprocessing = {
    'mmc': ['Garage Square Feet', 'Land Price', 'County factor'],
    'le': ['Total Property Area', ]
}

# special_discount_preprocessing = {
#     'mmc': ['Special Discount', 'VPC Discount', 'Insurance'],
#     'le': []
# }

filename = "Datasets/elementary_dataset_new.csv"
OUTPUT_VAR_NAME = get_outVar(filename)
minVal, maxVal = min(pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME]), max(
    pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME])
train_regression_model_on_data(filename, elementary_data_preprocessing, "elementary_")

filename = "Datasets/firebonus_dataset_new.csv"
OUTPUT_VAR_NAME = get_outVar(filename)
minVal, maxVal = min(pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME]), max(
    pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME])
train_regression_model_on_data(filename, firebonus_data_preprocessing, "firebonus_")
#
# filename = "Datasets/special_discount_new.csv"
# data = pd.read_csv(filename, sep=",")
# OUTPUT_VAR_NAME = get_outVar(filename)
# minVal, maxVal = min(pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME]), max(
#     pd.read_csv(filename, sep=",")[OUTPUT_VAR_NAME])
# train_regression_model_on_data(filename, special_discount_preprocessing, "special_discount_")

print("Writing results in file...")
with open("resultFile.txt", 'w+') as f:
    f.write(str(RESULTS))
print("Written results")
