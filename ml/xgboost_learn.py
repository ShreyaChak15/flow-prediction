import xgboost
import os
import xgboost_util
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import logging
import matplotlib.pyplot as plt
import timeit

NUMBER_OF_TREES = 50
WINDOW_SIZE = 5

TARGET_COLUMN = 'flow_size'


def print_performance(files, MODEL_SAVE_PATH, scaling, model, op, write_to_simulator=False):
    real = []
    predicted = []
    for f in files:
        data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN)
        inputs, outputs = xgboost_util.make_io(data)

        model = pickle.load(open(MODEL_SAVE_PATH, "rb"))
        y_pred = model.predict(xgboost.DMatrix(inputs, feature_names=data[0][0].columns))
        pred = y_pred.tolist()

        real += outputs
        predicted += pred

    xgboost_util.print_metrics(real, predicted, op)


def main(TEST_NAME, output_file):
    random.seed(0)

    TRAINING_PATH = 'data/ml/' + TEST_NAME + '/training/'
    TEST_PATH = 'data/ml/' + TEST_NAME + '/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME + '/validation/'
    MODEL_SAVE_PATH = 'model/xgboost/model_' + TEST_NAME + '.pkl'
    # LOG_PATH = 'model/xgboost/log_' + TEST_NAME + '.log'
    PLOT_PATH = 'results/xgboost/' + TEST_NAME

    # logging.basicConfig(level=logging.DEBUG)
    # file_handler = logging.FileHandler(LOG_PATH)

    training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
    test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
    validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

    scaling = xgboost_util.calculate_scaling(training_files)
    data = xgboost_util.prepare_files(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN)

    inputs, outputs = xgboost_util.make_io(data)

    # fit model no training data
    param = {
        'num_epochs': NUMBER_OF_TREES,
        'max_depth': 10,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'base_score': 2,
        'silent': 1,
        'eval_metric': 'mae'
    }

    training = xgboost.DMatrix(inputs, outputs, feature_names=data[0][0].columns)
    print(len(outputs))
    print('Training started')
    model = xgboost.train(param, training, param['num_epochs'])
    pickle.dump(model, open(MODEL_SAVE_PATH, "wb"))

    show_plots(MODEL_SAVE_PATH)

    xgboost.plot_importance(model, max_num_features=10)
    # plt.rcParams['figure.figsize'] = [50, 10]
    plt.title('XGBoost Feature importance plot for case ' + TEST_NAME)
    plt.savefig(os.path.join(PLOT_PATH, 'feature_importance.png'))
    plt.show()

    print('TRAINING')
    output_file.write('TRAINING\n')
    print_performance(training_files, MODEL_SAVE_PATH, scaling, model, output_file)

    print('TEST')
    output_file.write('TEST\n')
    print_performance(test_files, MODEL_SAVE_PATH, scaling, model, output_file)

    print('VALIDATION')
    output_file.write('VALIDATION\n\n')
    print_performance(validation_files, MODEL_SAVE_PATH, scaling, model, output_file)


if __name__ == "__main__":
    RESULTS_PATH = 'results/xgboost'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')

    print("Running all experiments:\n")
    for test_name in ["KMeans", "PageRank", "SGD", "tensorflow", "web_server"]:
        print("Case %s" % (test_name))
        output_file.write('CASE: ' + test_name + '\n')
        main(test_name, output_file)
    output_file.close()