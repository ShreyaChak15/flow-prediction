#!bin/bash

echo "Testing the models for prediction"

echo "\n\n Feed Forward Neural Network"
python ml/ffnn.py

echo "\n\n Long Short Term Memory"
python ml/lstm.py

echo "\n\n XGBoost"
python ml/xgboost_learn.py