# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:22:46 2019

@author: Harsh Anand
"""

import pickle

with open('picklefile.pkl','rb') as f:
    reg = pickle.load(f)
    
reg.predict([[4,3,2]])