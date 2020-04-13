# Main paper
[https://www.usenix.org/system/files/nsdi19-dukic.pdf] NSDI 19: Is advance knowledge of flow sizes a plausible assumption?

## Traces are obtained from the following applications:
- KMeans
- PageRank
- SGD
All implemented on Spark clusters
- Tensorflow
- Web Workload

## Machine Learning models implemented:
- Feed Forward Neural Networks
- Long Short Term Memory
- Gradient Boosting Decision Trees
- Convolutional Neural Networks (yet to test)

## To run the Machine Learning models
(Codes are written in Python3 - can be found in the `ml` directory)

From base directory, run `python ml/name_of_ml model.py`

e.g `python ml/xgboost_learn.py`

## Results
- The values for different error metrics are stored in:
`results/name_of_ml_model/results.txt`
- The log files for FFNN and LSTM are stored in:
`results/name_of_ml_model/loss_models`
- Loss plots for FFNN and LSTM are stored in:
`results/name_of_ml_model/loss_models`
- The models are stored in:
`model/name_of_ml_model`

We validate the paper's claim that XGBoost gives the faster convergence and good values of R2
for practical use.

We then try to gauge which features might be important for each application.
Plots for feature importance alongside their F2 scores are stored in:
`results/xgboost/name_of_application`

**Note** : Due to the large size of the files, Git open downloads pointers instead of the original files. You can use `git lfs` to download the complete files.
