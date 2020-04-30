#!bin/bash

echo "Training the models for prediction"

echo "\n\n Feed Forward Neural Network"
python ml/ffnn.py -train

echo "\n\nLong Short Term Memory"
python ml/lstm.py -train

echo "\n\nXGBoost"
python ml/xgboost_learn.py -train