#!/bin/bash

echo "Testing the models for prediction"

echo "Feed Forward Neural Network"
python ml/ffnn.py

echo "Long Short Term Memory"
python ml/lstm.py

echo "XGBoost"
python ml/xgboost_learn.py
