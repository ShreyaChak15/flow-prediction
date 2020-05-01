#!/bin/bash
echo "Testing the models for prediction"

echo -e "\n\n++++++++++++Feed Forward Neural Network++++++++++++"
python ml/ffnn.py

echo -e "\n\n++++++++++++Long Short Term Memory++++++++++++"
python ml/lstm.py

echo -e "\n\n++++++++++++XGBoost++++++++++++"
python ml/xgboost_learn.py
