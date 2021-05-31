# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:02:34 2021

@author: Zhiyun Gong
"""


import streamlit as st
from st_aggrid import AgGrid
import SessionState
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modAL.models import BayesianOptimizer, ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import max_UCB, max_EI, optimizer_EI, optimizer_UCB
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.graph_objects as go


def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

# Teach model on new data
def teach_model(model,X,y):
    model.teach(X,y)
    
    
# Get new designs
def get_design_BO(model, pool, batch_size):
    scaler = MinMaxScaler().fit(pool)
    exp_pool_df_norm = scaler.transform(pool)

    utilities = pd.DataFrame(optimizer_UCB(model, exp_pool_df_norm),columns=['Utility']).sort_values(by=['Utility'], ascending=False).round(4)
    util_cutoff = utilities['Utility'].to_list()[batch_size]
    utilities['top'] = utilities['Utility'] > util_cutoff
   
    pca = PCA(n_components=2)
    new_df_reduced = pd.DataFrame(pca.fit_transform(pool),columns=['PC1','PC2']).join(utilities, how='right')
    
    # session_state.util_plot = px.scatter_3d(new_df_reduced, x='PC1', y='PC2', z='Utility', color = 'top')
    util_plot = px.scatter(new_df_reduced, x='PC1', y='PC2', color = 'Utility')

    new_batch_idx = new_df_reduced.loc[new_df_reduced['top']==True,:].index.to_list()
    new_batch_df = pool.iloc[new_batch_idx,:].join(utilities).drop(columns = ['top'])

    return util_plot, new_batch_idx, new_batch_df



# Update evaluated data and unevaluated pool
def update_data(new_exp,new_batch_idx, exp_hist, pool, model):
    
    exp_hist = pd.concat([exp_hist,new_exp]).reset_index(drop=True)
    # hist_tbl.add_rows(new_exp)
    pool = pool.drop(new_batch_idx, axis=0)

    
    #Fit model
    teach_model(model, new_exp.iloc[:,:-1],new_exp[['Objective']].to_numpy().reshape(-1))
    
    return exp_hist, pool, model.get_max()[1]



def main():
    st.set_page_config(layout="wide")
    session_state = SessionState.get(data_ready = False, model_init=False, batch_size = 1, exp_hist =None,
                                      pool = None, curr_perf = 0, new_batch = None, show_design = False)

    st.title("Active learning hub")
    c1, c2, c3 = st.beta_columns((1,2,2))
    
    st.sidebar.markdown('## Experiment settings')
    task_type = st.sidebar.selectbox('Task type:', ['Objective optimization', 'Regression model training'])
    c1.subheader('Performing '+task_type)
    

    if task_type == 'Objective optimization':
        # c1.subheader('Current best objective: ' + str(session_state.curr_perf))
        # st.sidebar.selectbox('Utility function',['Expected Improvement',
        #                                           'Upper Confidence Bound',
        #                                           'Probability of Improvement'])
        # ---------------- Specify the number of parameters ----------------------
        num_params = st.sidebar.number_input('Number of design parameters:',1)
        
    if task_type == 'Regression model training':
        c1.subheader('Current r^2: ' + str(session_state.curr_perf))
        reg_type = st.sidebar.selectbox('Model type',['Linear regressor',
                                            'Random forest regressor'])
        num_params = st.sidebar.number_input('Number of independent variables:',2)


    
    
    
    
    param_names = []

    for i in range(num_params):
        st.sidebar.subheader('Parameter setting:')
        param_names.append(st.sidebar.text_input('Parameter name '+ str(i+1)))

    
        
    # ------------------- Upload evaluated experiments to initialize the model ------------------
    exp_history = st.sidebar.file_uploader('Upload evaluated experiments', type=['txt','csv'])
    if exp_history is not None and session_state.model_init == False:
        exp_hist_df = pd.read_csv(exp_history.name,names = param_names + ['Objective'],  header = 0, index_col=False)

        session_state.exp_hist = exp_hist_df

        st.sidebar.info('Experiments uploaded!')

    else:
        st.sidebar.warning('Please upload initial experiments')
    
    # ------------------- Upload available pool ------------------
    exp_pool_file = st.sidebar.file_uploader('Upload a .csv file containing unlabeled instances', type = ['txt','csv'])
    if exp_pool_file is not None:
        exp_pool_df = pd.read_csv(exp_pool_file, names = param_names, header = 0, index_col=False)

        st.sidebar.info('Pool uploaded! Containing ' + str(len(exp_pool_df))+' instances')
        if session_state.model_init ==False:
            session_state.pool = exp_pool_df

    if exp_pool_file is not None and exp_history is not None:
        session_state.data_ready = True
        
    
    #---------------------- Train the model using initial data-----------------
        
        
    init_btn = st.sidebar.button('Fit the model!')
    if init_btn and exp_history is not None:
        X_training = session_state.exp_hist[param_names]
        y_training = session_state.exp_hist[['Objective']]

    #------------------ Initialize corresponding models --------------------------
        #----------------- Regression model training -----------
        if task_type == 'Regression model training':
            print("regression")
            scaler = MinMaxScaler().fit(X_training)
            X_training = scaler.transform(X_training)
            
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
            
            AL_model = ActiveLearner(
                estimator=GaussianProcessRegressor(kernel=kernel),
                query_strategy=GP_regression_std,
                X_training = X_training,
                y_training = y_training.to_numpy().reshape(-1)
                )
            
            session_state.model = AL_model
            session_state.model_init = True
            y_pred, y_std = AL_model.predict(X_training, return_std=True)
            
            session_state.curr_perf = r2_score(y_training, y_pred)
                
                
        # ---------------- Optimization ------------------
        if task_type == 'Objective optimization':
            # session_state.curr_perf = np.max(session_state.exp_hist[['Objective']])
            scaler = MinMaxScaler().fit(X_training)
            X_training = scaler.transform(X_training)
            
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e5))

            regressor = GaussianProcessRegressor(kernel = kernel)
        
            BO_model = BayesianOptimizer(
                estimator = regressor)
            teach_model(BO_model, X = X_training,
                y = y_training.to_numpy().reshape(-1))
            # st.info("Bayesian Optimizer")
            session_state.model = BO_model
            session_state.model_init = True
            session_state.curr_perf = BO_model.get_max()[1]
            st.info('Model initialized')
    
    


        
    with c1:
        

        session_state.batch_size = st.slider('Batch size', 1, 100)
        st.markdown('#### Size of next batch:' + str(session_state.batch_size))
        get_design = st.button('Get a new batch of designs')
    
    
    if get_design:
        # print(len(session_state.pool))
        
        session_state.util_plot, session_state.new_batch_idx, session_state.new_batch = get_design_BO(session_state.model,
                                                               session_state.pool, 
                                                               session_state.batch_size)
        session_state.show_design = True
        
        
        
        
    if session_state.show_design:
       
        with c2:
            st.plotly_chart(session_state.util_plot, use_container_width=True)
        
        

        with c3:
            st.subheader('Next batch experiment designs')
            
            st.dataframe(session_state.new_batch)
            new_obj_str = st.text_input('Objective values for the new batch of experiments:')
            

            update = st.button('Update evaluated and unlabeled pool')
            # Update model and data
            if update:
                new_obj_list = list(map(float, new_obj_str.split(',')))
                new_exp = pd.concat([session_state.new_batch.iloc[:,:num_params],pd.DataFrame(new_obj_list, index = session_state.new_batch_idx, columns=['Objective'])],axis=1).reset_index(drop=True)
                session_state.exp_hist, session_state.pool, session_state.curr_perf =update_data(new_exp,session_state.new_batch_idx, session_state.exp_hist, 
                                                     session_state.pool, session_state.model)
                # session_state.exp_hist = pd.concat([session_state.exp_hist,new_exp]).reset_index(drop=True)
                # # hist_tbl.add_rows(new_exp)
                # session_state.pool = session_state.pool.drop(session_state.new_batch_idx, axis=0)
                session_state.new_batch = None
                session_state.new_batch_idx = None
                
                # #Fit model
                # teach_model(session_state.model, new_exp.iloc[:,:num_params],new_exp[['Objective']].to_numpy().reshape(-1))
                # # session_state.model.teach()
                # session_state.curr_perf = session_state.model.get_max()[1]
                st.write(len(session_state.exp_hist))
                print('After update'+str(len(session_state.pool)))
                st.write(len(session_state.pool))
                st.write(session_state.curr_perf)
                session_state.show_design =False


            # st.write(new_obj_list)
    with c1:
        hist_tbl = st.dataframe(session_state.exp_hist)

    if task_type == 'Objective optimization':
        
            c1.subheader('Current best objective: ' + str(session_state.curr_perf))

            
main()
