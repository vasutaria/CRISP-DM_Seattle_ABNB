import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
from sklearn.svm import SVC #(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier #(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

%matplotlib inline

Project: This project is Seattle ABNB Data from 2016.  Analyzing the 2016 market data to make a decision in investing in Seattle.

The Dataset can be found https://www.kaggle.com/airbnb/seattle/data
Dataset consists of 3 Files:
Calendar - Listings of all available and not available properties in 2016
Listing - 3818 records of listings
Review - With details review comments 

Summary:
This project helps answer 3 questions about Seattle market based on 2016 ABNB Data.  
1)What is the most popular price and which neighbourhood had the highest price. $150 was the most popular price and Rosevelt neighbourhood had the highest price.
2)What time of the year is the ABNB most profitable? Seems like the 1st half of the year is better specifically Mid January to March and May, June and July.
3)What property type generates the highest revenue?  House and Apartment property types generate the most revenue.


