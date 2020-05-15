# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:39:39 2020

@author: Drones
"""


import pickle
import tensorflow as tf
import numpy as np


with open("model.cpickle", "rb") as f:
    
    model = pickle.load(f)
    


np.savetxt("coeff.txt", model.coef_, delimiter=",")
np.savetxt("intercept.txt", model.intercept_, delimiter=",")



np