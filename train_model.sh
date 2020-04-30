#!/bin/bash

echo "Training the models for prediction"

echo "Feed Forward Neural Network"
python ml/ffnn.py -train

echo "Long Short Term Memory"
python ml/lstm.py -train

echo "XGBoost"
python ml/xgboost_learn.py -train
