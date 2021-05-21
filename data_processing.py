# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:56:44 2021

@author: Zhiyun Gong
"""


import pandas as pd
import numpy as np
import seaborn as sns
import copy
import random
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import max_UCB, max_EI, optimizer_EI, optimizer_UCB
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import plotly.express as pxs
from sklearn.decomposition import PCA



# Read in data
df = pd.read_excel('protein_evo.xlsx')[['Variants','Fitness']]
X = df['Variants'].str.split('',expand=True).iloc[:,1:5]
X.columns = ['loc' + str(i+1) for i in range(4)]
y = df[['Fitness']]




# Encode amino acids with integers from 1 to 20
alphabet = np.sort(X.loc1.unique())
X_int = copy.deepcopy(X)
for a in alphabet:
    X_int=X_int.replace(a,str(np.where(alphabet == a)[0][0]))
X_int = X_int.astype(int)

pca = PCA(n_components=2)
components = pca.fit_transform(X_int)
pd.DataFrame(components).plot().scatter(x=0, y=1)
# X_reduced =pd.DataFrame(pca.fit_transform(X_int),columns = ['PC1','PC2'])

fig = px.scatter(components, x=0, y=1)
fig.show()

# fig = px.scatter(
#     X_reduced
#     )
# fig.show()

# Normalization
X_pool = copy.deepcopy(X_int)
scaler = MinMaxScaler().fit(X_pool)
X_pool = scaler.transform(X_pool)
y_pool = copy.deepcopy(y).to_numpy()

# Randomly select 10 instances as initial samples
random.seed(10)
training_idx = np.random.choice(len(X_pool),100, replace=False)
X_train = X_pool[training_idx,:]
y_train = y_pool[training_idx]
np.max(y_train)

# training_data = pd.concat([X_int.iloc[training_idx,:],y_train], axis=1)
# training_data.to_csv('training.csv', ',', index=False)
    
# Put remaining instances into a pool
# X_pool = X_pool.drop(training_idx, axis=0)
# y_pool = y_pool.drop(training_idx, axis=0)
X_pool = np.delete(X_pool, training_idx, axis=0)
y_pool = np.delete(y_pool, training_idx)

# X_int.drop(training_idx).to_csv('init_pool_X.csv',",",index=False)
# init_pool = pd.concat([X_int.drop(training_idx),y_pool], axis=1)
# init_pool.to_csv('init_pool.csv',",",index=False)



# Model initialization
kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel = kernel)
optimizer = BayesianOptimizer(
    estimator = regressor,
    X_training = X_train,
    y_training = y_train.reshape(-1),
    query_strategy = max_EI)

_, y_max = optimizer.get_max()
y_max

batch_size = 100
query_idx, X_query = optimizer.query(X_pool, n_instances = batch_size)
utilities = optimizer_UCB(optimizer, X_pool)

np.max(utilities)
np.min(utilities)

y_query = y_pool[query_idx]
# X_pool = X_pool.drop(query_idx, axis=0)
# y_pool = y_pool.drop(query_idx, axis=0)
X_pool = np.delete(X_pool, query_idx, axis=0)
y_pool = np.delete(y_pool, query_idx)


optimizer.teach(X_query, y_query.reshape(-1))

X_max, y_max = optimizer.get_max()
y_max



sns.histplot(utilities)


tsne = TSNE(n_components = 2, random_state = 0)
projections = tsne.fit_transform(X_int)

fig = px.scatter(
    projections, x=0, y = 1)
st.plotly_chart(fig)


