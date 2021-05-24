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
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import max_UCB, max_EI, optimizer_EI, optimizer_UCB
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go



def main():
    st.set_page_config(layout="wide")

    st.title("Active learning hub")
    c1, c2, c3 = st.beta_columns((1,2,2))
    session_state = SessionState.get(data_ready = False, model_init=False, batch_size = 1, pool = None)
    
    st.sidebar.markdown('## Experiment settings')
    task_type = st.sidebar.selectbox('Task type:', ['Objective optimization', 'Regression model training'])
    

    
    # ---------------- Specify the number of parameters ----------------------
    num_params = st.sidebar.number_input('Number of design parameters:',1)
    
    
    param_names = []
    # params_labels = []
    # lbs = []
    # ubs = []
    for i in range(num_params):
        # st.subheader('Parameter setting:')
        param_names.append(st.sidebar.text_input('Parameter name '+ str(i+1)))
        # exec("range_bin" +str(i) + "= st.checkbox('Integer list')")
        # x = exec("range_bin" + str(i))
        # st.write(x)
        
        # lbs.append(st.number_input('Lower bound' +str(i+1),1))
        # ubs.append(st.number_input('Upper bound'+str(i+1),5))
        # labels = range(lbs[i],ubs[i])
        # params_labels.append(labels)
        # params_labels.append(st.text_input('Legit values for parameter'+str(i+1)).split(','))
    
        
    # ------------------- Upload evaluated experiments to initialize the model ------------------
    exp_history = st.sidebar.file_uploader('Upload evaluated experiments', type=['txt','csv'])
    if exp_history is not None:
        exp_hist_df = pd.read_csv(exp_history.name,names = param_names + ['Objective'],  header = 0, index_col=False)
        X_training = exp_hist_df[param_names]
        y_training = exp_hist_df[['Objective']]

        st.sidebar.info('Experiments uploaded!')
        with c1:
            show_hist = st.checkbox('Show evaluated experiments')
            if show_hist:
                # AgGrid(exp_hist_df, editable=True)
                st.dataframe(exp_hist_df)
        session_state.exp_hist = X_training
    else:
        st.sidebar.warning('Please upload initial experiments')
    
    # Train the model using initial data
    init_btn = st.sidebar.button('Fit the model!')
    if init_btn and exp_history is not None:
                

            
    #------------------ Initialize corresponding models --------------------------
        if task_type == 'Objective optimization':
            scaler = MinMaxScaler().fit(X_training)
            X_training = scaler.transform(X_training)
            
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e5))

            regressor = GaussianProcessRegressor(kernel = kernel)
        
            BO_model = BayesianOptimizer(
                estimator = regressor,
                query_strategy = max_UCB,
                X_training = X_training,
                y_training = y_training.to_numpy().reshape(-1))
            st.info("Bayesian Optimizer")
            session_state.model = BO_model

        # session_state.model.teach(X_training, y_training.to_numpy().reshape(-1))
        # session_state.model = BO_model
        st.write(session_state.model.get_max())
        session_state.model_init = True
        st.info('Model initialized')
    
    
    # ------------------- Upload available pool ------------------
    exp_pool_file = st.sidebar.file_uploader('Upload a .csv file containing unlabeled instances', type = ['txt','csv'])
    if exp_pool_file is not None:
        exp_pool_df = pd.read_csv(exp_pool_file, names = param_names, header = 0, index_col=False)
        # norm_pool = st.sidebar.checkbox("Need to be Normalized?")
        # if norm_pool ==True:
        #     scaler = MinMaxScaler().fit(exp_pool_df)
        #     exp_pool_df = scaler.transform(exp_pool_df)
        #     st.sidebar.info("Data normalized!")
        st.sidebar.info('Pool uploaded! Containing ' + str(len(exp_pool_df))+' instances')
        # st.write(exp_pool_df) 
        session_state.pool = exp_pool_df

    if exp_pool_file is not None and exp_history is not None:
        session_state.data_ready = True
    
   
     
    if session_state.model_init == True and session_state.pool is not None:    
        # Get suggested designs
        # session_state.batch_size = st.sidebar.number_input('Batch size',1)
        
        
        with c1:
            session_state.batch_size = st.slider('Batch size', 1, 100)
            st.markdown('#### Size of next batch:' + str(session_state.batch_size))
            get_design = st.button('Get a new batch of designs')
        
        
        if get_design:
            # st.write('HHHH')
            # query_idx, X_query = BO_model.query(exp_pool_df, n_instances = session_state.batch_size)
            # st.write(X_query)
            scaler = MinMaxScaler().fit(session_state.pool)
            exp_pool_df_norm = scaler.transform(session_state.pool)

            utilities = pd.DataFrame(optimizer_UCB(session_state.model, exp_pool_df_norm),columns=['Utility']).sort_values(by=['Utility'], ascending=False).round(4)
            util_cutoff = utilities['Utility'].to_list()[session_state.batch_size]
            utilities['top'] = utilities['Utility'] >= util_cutoff
            # utilities_sorted = utilities.iloc[:session_state.batch_size,:]
            # new_df = session_state.pool.join(utilities_sorted, how='right')
            pca = PCA(n_components=2)
            new_df_reduced = pd.DataFrame(pca.fit_transform(session_state.pool),columns=['PC1','PC2']).join(utilities, how='right')
            
            # fig_surf = go.Figure(data=[go.Surface(z=new_df_reduced.Utility,x = new_df_reduced.PC1, y=new_df_reduced.PC2)])
            # st.plotly_chart(fig_surf)
            # with c1: 
            fig_1 = px.scatter_3d(new_df_reduced, x='PC1', y='PC2', z='Utility', color = 'top')
            with c2:
                st.plotly_chart(fig_1, width =100)
            
            
            new_batch_idx = new_df_reduced.loc[new_df_reduced['top']==True,:].index.to_list()
            new_batch_df = session_state.pool.iloc[new_batch_idx,:].join(utilities).drop(columns = ['top'])
            with c3:
                st.dataframe(new_batch_df)
            
main()
