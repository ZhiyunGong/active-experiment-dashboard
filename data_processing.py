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
from modAL.models import BayesianOptimizer, ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import max_UCB, max_EI, optimizer_EI, optimizer_UCB
from sklearn.gaussian_process.kernels import Matern,RBF,WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import plotly.express as pxs
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import r2_score


#----------------------- Protein directed evolution ------------------------------

# Read in data
df = pd.read_excel('protein_evo.xlsx')[['Variants','Fitness']]
X = df['Variants'].str.split('',expand=True).iloc[:,1:5]
X.columns = ['loc' + str(i+1) for i in range(4)]
y = df[['Fitness']]


y_max = []
for i in range(100):
    
    idx = np.random.choice(len(y),200)
    y_max.append(np.max(y.iloc[idx,:]))
    

np.median(y_max)



    
    
# Encode amino acids with integers from 1 to 20
alphabet = np.sort(X.loc1.unique())
X_int = copy.deepcopy(X)
for a in alphabet:
    X_int=X_int.replace(a,str(np.where(alphabet == a)[0][0]))
X_int = X_int.astype(int)


# Normalization
X_pool = copy.deepcopy(X_int)
scaler = MinMaxScaler().fit(X_pool)
X_pool = scaler.transform(X_pool)
y_pool = copy.deepcopy(y).to_numpy()

# Randomly select 10 instances as initial samples
np.random.seed(222)
training_idx = np.random.choice(len(X_pool),100, replace=False)
# X_train = X_pool.iloc[training_idx,:].to_numpy()
X_train = X_pool[training_idx,:]
y_train = y_pool[training_idx]
# Max objective in the initial training set
np.max(y_train)

training_data = pd.concat([X_int.iloc[training_idx,:].reset_index(drop=True),pd.DataFrame(y_train)], axis=1)
training_data.to_csv('training.csv', ',', index=False)
    
# Put remaining instances into a pool
# X_pool = X_pool.drop(training_idx, axis=0)
# y_pool = y_pool.drop(training_idx, axis=0)
X_pool = np.delete(X_pool, training_idx, axis=0)
y_pool = np.delete(y_pool, training_idx)

X_int.drop(training_idx).to_csv('init_pool_X.csv',",",index=False)
init_pool = pd.concat([X_int.drop(training_idx).reset_index(drop=True),pd.DataFrame(y_pool)], axis=1)
init_pool.to_csv('init_pool.csv',",",index=False)



# Model initialization
kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e5))
                
regressor = GaussianProcessRegressor(kernel = kernel)
optimizer = BayesianOptimizer(
    estimator = regressor,
    X_training = X_train,
    y_training = y_train.reshape(-1),
    query_strategy = max_UCB)


#--------------- Bayesian Optimization ---------------
batch_1_idx = [142704, 142722, 142686, 142685, 142703, 142723, 142705, 142721, 142687, 142741, 143051, 142740, 143050, 143069, 143032, 143033, 142724, 143070, 142706, 143052]
y_pool[batch_1_idx,]
# 1.19600001e-02, 8.06127455e-03, 8.23073078e-03, 5.60784317e-03,
#        2.49166668e-03, 4.07862105e-03, 2.02422574e+00, 3.63827436e-03,
#        8.66630046e-01, 1.21814815e-03, 2.81490993e-01, 5.33928574e-03,
#        5.89426526e-03, 2.25715687e-02, 1.04082279e-02, 5.08380379e-02,
#        2.87500002e-03, 0.00000000e+00, 8.28046329e-03, 9.57388315e-01

batch_2_idx = [143068, 142742, 142684, 135969, 135949, 142739, 135950, 143087, 135968, 142688, 143034, 143088, 135988, 142760, 142702, 135948, 135970, 135989, 143031, 27203]
y_pool[batch_2_idx,]

# 1.33699188e-02, 3.39772729e-03, 3.58480947e-01, 2.52844405e-02,
#        2.43870491e-03, 3.24358976e-03, 3.13467744e-02, 3.04537039e-02,
#        8.78199832e-03, 6.69662598e-03, 1.01025758e+00, 2.74083335e-02,
#        2.98457352e-03, 3.22213962e+00, 4.65036289e-01, 2.91578016e-02,
#        2.58004457e+00, 1.97064111e-03, 1.75610280e-01, 8.10897440e-03


batch_3_idx = [27204, 27185, 142761, 27222, 135987, 142759, 143071, 135951, 135967, 143086, 27186, 142683, 136310, 27221, 143053, 136330, 143089, 136329, 142725, 143049]
y_pool[batch_3_idx,]
# 8.00243313e-03, 0.00000000e+00, 5.43228231e-03, 1.69187244e-03,
#        2.51914830e-03, 6.31112942e-03, 4.24935403e-03, 5.86639161e-01,
#        9.13611116e-02, 0.00000000e+00, 5.85648151e-03, 3.93922168e+00,
#        1.17632333e-02, 0.00000000e+00, 1.27905556e-01, 3.01131032e-01,
#        1.35349795e-02, 1.02052306e-02, 1.40555556e-03, 4.35459504e-01




batch_4_idx = [136008, 136007, 142743, 136311, 27223, 136348, 135947, 136309, 27205, 142707, 27240, 142720, 143035, 135990, 136347, 27184, 27241, 143030, 136328, 135971]
y_pool[batch_4_idx,]

# 2.38492862e-03, 2.18829010e-03, 0.00000000e+00, 7.78208533e-02,
#        3.28900002e-03, 9.97876219e-03, 3.78293629e-01, 5.48166670e-02,
#        1.08636667e+00, 1.90817512e-01, 0.00000000e+00, 1.85577258e-02,
#        7.66666671e-03, 1.43031094e-03, 4.24935403e-03, 2.68489797e-01,
#        0.00000000e+00, 3.35752085e+00, 0.00000000e+00, 7.75122553e-03

batch_5_idx = [143105, 136349, 136331, 143106, 135986, 27187, 142762, 142701, 143107, 142689, 135966, 142758, 136006, 19535, 136009, 136308, 27119, 27239, 136312, 135952]
y_pool[batch_5_idx,]
# 4.98333336e-02, 6.93004639e-03, 1.44627000e+00, 3.36232167e+00,
#        0.00000000e+00, 4.29322947e-01, 7.90978751e-02, 4.43422929e+00,
#        1.93470589e-02, 3.09015525e-02, 1.18141007e+00, 9.98062564e-03,
#        3.84678365e-03, 4.72557474e-03, 1.51010102e-03, 2.03933334e-01,
#        2.75658526e-02, 2.12467701e-03, 6.58270757e-01, 3.56724514e-03


# X_rest = np.delete(X_pool, queried_idx, axis=0)
# y_rest = np.delete(y_pool, queried_idx)

# optimizer.teach(X_queried,y_queried.reshape(-1))

# y_rest_pred = optimizer.predict(X_rest)
# r2_score(y_rest, y_rest_pred)


# y_queried_pred = optimizer.predict(optimizer.X_training)
# r2_score(optimizer.y_training, y_queried_pred)


# ------------------------- Buffer composition data -------------------------------
df2 = pd.read_csv('DataPool.csv').iloc[:-3,:]
np.random.seed(111)
# testing_idx = np.random.choice(len(df2),507, replace=False)
# df2_training = df2.drop(testing_idx,axis=0).reset_index(drop=True)
init_idx = np.random.choice(len(df2),50, replace=False)
df2_init = df2.iloc[init_idx,:].rename(columns = {"Yield": "Objective"})
df2_init.to_csv('buffer_training.csv',',',index=False)
X_training = df2_init.iloc[:,:-1]
y_training = df2_init[['Objective']]

df2_pool = df2.drop(init_idx,axis=0).rename(columns = {"Yield": "Objective"})
X_pool = df2_pool.drop('Objective',axis=1)
X_pool.to_csv('buffer_init_pool_X.csv',',',index=False)
y_pool = df2_pool[['Objective']]



def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


# #--------------- Bayesian Optimization ---------------
# batch_1_idx = [271, 125, 87, 462, 70, 467, 141, 365, 112, 105, 65, 307, 344, 454, 35, 456, 459, 361, 207, 474, 91, 114, 299, 495, 447, 132, 234, 246, 320, 145, 334, 172, 296, 347, 442, 415, 155, 174, 127, 315, 55, 458, 252, 32, 215, 75, 144, 455, 494, 478]
# y_pool.loc[batch_1_idx,:].to_numpy().reshape(1,-1)

# batch_2_idx = [0, 373, 339, 338, 337, 336, 335, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 319, 318, 317, 316, 314, 313, 312, 340, 341, 342, 358, 371, 370, 369, 368, 367, 366, 364, 363, 362, 360, 359, 357, 343, 356, 355, 354, 353, 352, 351]
# y_pool.iloc[batch_2_idx,:].to_numpy().reshape(1,-1)


# batch_3_idx = [1, 290, 302, 301, 300, 298, 297, 295, 294, 293, 292, 291, 289, 2, 288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 303, 304, 305, 306, 384, 383, 382, 381, 380, 379, 378, 377, 376, 375, 374, 372, 350, 349, 348, 346, 345, 311, 310, 309, 308, 278]
# y_pool.iloc[batch_3_idx,:].to_numpy().reshape(1,-1)


# batch_4_idx = [3, 258, 266, 265, 264, 263, 262, 261, 260, 259, 257, 268, 256, 255, 254, 253, 251, 250, 249, 248, 267, 269, 245, 388, 396, 395, 394, 393, 392, 391, 390, 389, 387, 270, 386, 385, 277, 276, 275, 274, 273, 272, 247, 244, 398, 210, 219, 218, 217, 216]
# y_pool.iloc[batch_4_idx,:].to_numpy().reshape(1,-1)


# batch_5_idx = [4, 238, 236, 235, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 237, 239, 214, 240, 409, 408, 407, 406, 405, 404, 403, 402, 401, 400, 399, 397, 243, 242, 241, 220, 213, 5, 192, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181]
# y_pool.iloc[batch_5_idx,:].to_numpy().reshape(1,-1)


#--------------- Active Regression ---------------
batch_1_idx = [0, 639, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 640, 654, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 625, 624, 623, 622, 596, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614]
# y_pool.iloc[batch_1_idx,:].to_numpy().reshape(1,-1)

batch_2_idx = [264, 96, 249, 918, 473, 383, 769, 9, 921, 672, 860, 548, 150, 471, 793, 49, 506, 233, 281, 941, 151, 374, 845, 800, 322, 935, 155, 537, 707, 311, 316, 45, 168, 553, 14, 467, 378, 414, 431, 111, 295, 531, 776, 585, 825, 274, 358, 133, 466, 749]
# y_pool.iloc[batch_2_idx,:].to_numpy().reshape(1,-1)


batch_3_idx = [488, 340, 127, 744, 314, 933, 900, 213, 743, 847, 940, 495, 350, 664, 457, 824, 292, 176, 256, 88, 165, 17, 275, 726, 132, 846, 328, 597, 765, 10, 734, 296, 41, 107, 348, 542, 745, 355, 903, 159, 175, 336, 237, 909, 300, 957, 309, 173, 56, 353]
# y_pool.iloc[batch_3_idx,:].to_numpy().reshape(1,-1)


batch_4_idx = [16, 234, 335, 885, 753, 250, 699, 510, 763, 521, 852, 938, 875, 563, 911, 323, 102, 873, 114, 951, 944, 439, 298, 835, 656, 670, 750, 57, 326, 441, 397, 819, 811, 166, 156, 3, 13, 352, 63, 509, 325, 859, 445, 449, 402, 489, 511, 840, 294, 72]
# y_pool.iloc[batch_4_idx,:].to_numpy().reshape(1,-1)

batch_5_idx = [34, 183, 273, 32, 818, 943, 43, 105, 844, 477, 809, 638, 526, 54, 857, 576, 708, 265, 816, 703, 157, 948, 167, 220, 80, 368, 514, 533, 81, 136, 549, 244, 919, 260, 788, 152, 254, 277, 95, 593, 331, 440, 409, 422, 204, 464, 931, 137, 62, 67]
# y_pool.iloc[batch_5_idx,:].to_numpy().reshape(1,-1)



# Model initialization
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
   + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
                
learner = ActiveLearner(
    estimator = GaussianProcessRegressor(kernel=kernel),
    X_training = X_training,
    y_training = y_training.to_numpy().reshape(-1),
    query_strategy = GP_regression_std)

y_pred = learner.predict(X_pool)
print(r2_score(y_pool, y_pred))
idx_list = [batch_1_idx, batch_2_idx, batch_3_idx, batch_4_idx, batch_5_idx]
drop_idx = []
for i in idx_list:
    drop_idx = drop_idx + i
    # print(drop_idx)
    X_query = X_pool.iloc[drop_idx,:]
    X_test = X_pool.reset_index(drop=True).drop(drop_idx,axis=0)
    y_test = y_pool.reset_index(drop=True).drop(drop_idx)
    y_query = y_pool.iloc[drop_idx,:].to_numpy().reshape(-1)
    
    learner.teach(X_query,y_query)
    y_pred = learner.predict(X_test)
    print(r2_score(y_test, y_pred))
    
    