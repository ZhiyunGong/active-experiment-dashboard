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
from sklearn.preprocessing import MinMaxScaler
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import max_UCB, max_EI
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def main():
    st.title("Active learning hub")
    session_state = SessionState.get(data_ready = False, model_init=False, batch_size = 1)
    
    st.sidebar.markdown('## Experiment settings')
    task_type = st.sidebar.selectbox('Task type:', ['Objective optimization', 'Regression model training'])
    #------------------ Initialize corresponding models --------------------------
    if task_type == 'Objective optimization':
        kernel = Matern()
        regressor = GaussianProcessRegressor(kernel = kernel)
    
        BO_model = BayesianOptimizer(
            estimator = regressor,
            query_strategy = max_EI)
        st.info("Bayesian Optimizer")
        session_state.model_init = True
        session_state.model = BO_model
    
    
    num_params = st.sidebar.number_input('Number of design parameters:',2)
    
    
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
    
    # st.write(param_names)
    # st.write(type(param_names))
    # st.write(params_labels)
        
    # ------------------- Upload evaluated experiments to initialize the model ------------------
    exp_history = st.sidebar.file_uploader('Upload evaluated experiments', type=['txt','csv'])
    if exp_history is not None:
        exp_hist_df = pd.read_csv(exp_history.name,names = param_names + ['Objective'],  header = 0, index_col=False)
        # st.write(exp_hist_df)1
        X_training = exp_hist_df[param_names]
        y_training = exp_hist_df[['Objective']]
        # norm = st.sidebar.checkbox("Need to be normalized?")
        # if norm ==True:
        #     scaler = MinMaxScaler().fit(X_training)
        #     X_training = scaler.transform(X_training)
        #     st.sidebar.info('Data normalized')
        st.sidebar.info('Experiments uploaded!')
        show_hist = st.checkbox('Show evaluated experiments')
        if show_hist:
            AgGrid(exp_hist_df, editable=True)
        # st.dataframe(X_training)
        session_state.exp_hist = X_training
    else:
        st.sidebar.warning('Please upload initial experiments')
    
    
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
        
        # tsne = TSNE(n_components = 2, random_state = 0)
        # projections = tsne.fit_transform(exp_pool_df)
    
        # fig = px.scatter(
        #     projections, x=0, y = 1,
        #     color = labels, labels = {'color':'Diagnosis'})
        # st.plotly_chart(fig)
        
    if exp_pool_file is not None and exp_history is not None:
        session_state.data_ready = True
    
    # Initialize Bayesian Optimizer
    init_btn = st.button('Initialize!')
    # hi = None
    if init_btn and session_state.data_ready:
        BO_model.teach(X_training, y_training.to_numpy().reshape(-1))
        st.info('Model initialized')
     
    if session_state.model_init == True:    
        # Get suggested designs
        session_state.batch_size = st.sidebar.number_input('Batch size',1)

        get_design = st.button('Get a new batch of designs')
        
        if get_design:
            st.write('HHHH')
            query_idx, X_query = BO_model.query(exp_pool_df, n_instances = session_state.batch_size)
            st.write(X_query)
    
main()