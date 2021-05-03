#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:09:51 2021

@author: ashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('Data/titanic_train.csv')
print(train.head())

test = pd.read_csv('Data/titanic_test.csv')
print(test.head())