# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:57:38 2020

@author: shreya
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

def show_plots(filename, dataset):
    hist = pickle.load(open(filename, "rb" ) )
    val_loss = hist['val_loss']
    loss = hist['loss']
    
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, max(len(val_loss), len(loss))+1, 5.0))
    plt.title(('Losses over epoch: ' + dataset))
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()

# plots

show_plots('../results/lstm/loss_models/model_KMeans.pkl', 'KMeans')

show_plots('../results/lstm/loss_models/model_PageRank.pkl', 'PageRank')

show_plots('../results/lstm/loss_models/model_SGD.pkl', 'SGD')

show_plots('../results/lstm/loss_models/model_web_server.pkl', 'Web Server')


show_plots('../results/ffnn/loss_models/model_KMeans.pkl', 'KMeans')

show_plots('../results/ffnn/loss_models/model_PageRank.pkl', 'PageRank')

show_plots('../results/ffnn/loss_models/model_SGD.pkl', 'SGD')

show_plots('../results/ffnn/loss_models/model_web_server.pkl', 'Web Server')
