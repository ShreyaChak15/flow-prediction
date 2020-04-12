# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:57:38 2020

@author: shreya
"""
import pickle
import matplotlib.pyplot as plt

def show_plots(filename):
    hist = pickle.load(open(filename, "rb" ) )
    val_loss = hist['val_loss']
    loss = hist['loss']
    
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(loss, label='Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('Val Loss')
    plt.title(('Losses over epoch'))
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
