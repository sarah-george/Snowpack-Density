
# coding: utf-8

# In[1]:


## Homework 4
# Import and display data from Google Sheets
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade -q gspread')
get_ipython().system('pip install gspread')
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install graphviz')

import pandas as pd
import pydotplus
import graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import sklearn
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_excel('Gauge_data.xlsx')
df2 = df


# # Regression Tree

# In[3]:


grouped_data = df2.groupby('density').mean()
grouped_data = grouped_data.reset_index(drop=False)

print(grouped_data)


# In[11]:


X_full = np.array(df2['gain']).reshape(-1,1)
y_full = np.array(df2['density']).reshape(-1,1)
X_means = np.array(grouped_data['gain']).reshape(-1,1)
y_means = np.array(grouped_data['density']).reshape(-1,1)

print(X_means)
print(y_means)
#print(y)


# In[6]:


# decision tree using gain to predict density
clf1 = tree.DecisionTreeRegressor(criterion='mse', max_depth=5)
clf1 = clf1.fit(X_full, y_full) #fit(X, y[, sample_weight, check_input, …])	Build a decision tree regressor from the training set (X, y).

# decision tree using density to predict gain
clf2 = tree.DecisionTreeRegressor(criterion='mse', max_depth=5)
clf2 = clf2.fit(y_full, X_full) #fit(X, y[, sample_weight, check_input, …])	Build a decision tree regressor from the training set (X, y).


#print(clf)
#print(clf.decision_path(X))


# In[7]:


dot_data = StringIO()
export_graphviz(clf1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.create_png()
Image(graph.create_png())


# In[8]:


dot_data = StringIO()
export_graphviz(clf2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.create_png()
Image(graph.create_png())


# In[12]:


#clf 1 (density predicting gain)

clf1_full_preds = clf1.predict(X_full) #predict(X[, check_input])	Predict class or regression value for X.=
print(clf1_full_preds)

print(clf1.score(X_means,y_means)) #score(X, y[, sample_weight])	Returns the coefficient of determination R^2 of the prediction.


#clf2 (density predicting gain)
clf2_full_preds = clf2.predict(y_full) #predict(X[, check_input])	Predict class or regression value for X.=
print(clf2_full_preds)

print(clf2.score(y_means,X_means)) #score(X, y[, sample_weight])	Returns the coefficient of determination R^2 of the prediction.

