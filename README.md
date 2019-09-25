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

