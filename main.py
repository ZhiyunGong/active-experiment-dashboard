# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:02:34 2021

@author: Zhiyun Gong
"""


import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modAL.models import BayesianOptimizer, ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from modAL.acquisition import optimizer_UCB
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import plotly.express as px

# ------------------- Initialize session state defaults -------------------
def init_session_state():
    defaults = {
        'data_ready': False,
        'model_init': False,
        'batch_size': 1,
        'exp_hist': None,
        'pool': None,
        'curr_perf': 0.0,
        'new_batch': None,
        'show_design': False,
        'no_iter': 0,
        'new_batch_idx': None,
        'util_plot': None,
        'model': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ------------------ Helper functions ------------------
def teach_model(model, X, y):
    model.teach(X, y)

def get_design_BO(model, pool, batch_size):
    scaler = MinMaxScaler().fit(pool)
    exp_pool_df_norm = scaler.transform(pool)

    utilities = pd.DataFrame(optimizer_UCB(model, exp_pool_df_norm), columns=['Utility'], index=pool.index).sort_values(by='Utility', ascending=False).round(4)
    pca = PCA(n_components=2)
    new_df_reduced = pd.DataFrame(pca.fit_transform(pool), columns=['PC1', 'PC2'], index=pool.index).join(utilities, how='right').reset_index(drop=True)
    util_plot = px.scatter(new_df_reduced, x='PC1', y='PC2', color='Utility')
    new_batch_idx = utilities.iloc[:batch_size].index.to_list()
    new_batch_df = pool.loc[new_batch_idx].join(utilities)
    return util_plot, new_batch_idx, new_batch_df

def GP_std(model, pool):
    _, std = model.predict(pool, return_std=True)
    return std

def get_design_AL(model, pool, batch_size):
    scaler = MinMaxScaler().fit(pool)
    exp_pool_df_norm = scaler.transform(pool)
    stds = pd.DataFrame(GP_std(model, pool), columns=['Uncertainty'], index=pool.index).sort_values(by='Uncertainty', ascending=False).round(4)
    pca = PCA(n_components=2)
    new_df_reduced = pd.DataFrame(pca.fit_transform(pool), columns=['PC1', 'PC2'], index=pool.index).join(stds, how='right').reset_index(drop=True)
    util_plot = px.scatter(new_df_reduced, x='PC1', y='PC2', color='Uncertainty')
    new_batch_idx = stds.iloc[:batch_size].index.to_list()
    new_batch_df = pool.loc[new_batch_idx].join(stds)
    return util_plot, new_batch_idx, new_batch_df

def update_data(new_exp, new_batch_idx, exp_hist, pool, model, task_type):
    exp_hist = pd.concat([exp_hist, new_exp]).reset_index(drop=True)
    pool = pool.drop(new_batch_idx, axis=0)
    teach_model(model, new_exp.iloc[:, :-1], new_exp[['Objective']].to_numpy().reshape(-1))
    if task_type == 'Objective optimization':
        curr_perf = model.get_max()[1]
    else:
        y_pred, y_std = model.predict(model.X_training, return_std=True)
        curr_perf = r2_score(model.y_training, y_pred)
    return exp_hist, pool, curr_perf

def check_param_names(exp_hist, pool):
    return exp_hist.columns.to_list()[:-1] == pool.columns.to_list()

# ------------------ Main ------------------
def main():
    init_session_state()
    st.set_page_config(layout="wide")
    st.title("Bayesian Optimization and Active Regression Dashboard")
    c1, c2, c3 = st.columns((1,2,2))

    st.sidebar.markdown('## Experiment settings')
    task_type = st.sidebar.selectbox('Task type:', ['Objective optimization', 'Regression model training'])
    c1.subheader(f'Performing {task_type}')
    param_names = []

    exp_history = st.sidebar.file_uploader('Upload evaluated experiments', type=['txt','csv'])
    if exp_history is not None and not st.session_state.model_init:
        st.session_state.exp_hist = pd.read_csv(exp_history, header=0, index_col=False)
        st.sidebar.info('Experiments uploaded!')
    else:
        st.sidebar.warning('Please upload initial experiments')

    exp_pool_file = st.sidebar.file_uploader('Upload a .csv file containing unlabeled instances', type=['txt','csv'])
    if exp_pool_file is not None:
        st.session_state.pool = pd.read_csv(exp_pool_file, header=0, index_col=False)
        st.sidebar.info(f'Pool uploaded! Containing {len(st.session_state.pool)} instances')

    if exp_history is not None and exp_pool_file is not None:
        st.session_state.data_ready = check_param_names(st.session_state.exp_hist, st.session_state.pool)

    if st.session_state.data_ready:
        param_names = st.session_state.pool.columns.to_list()

    init_btn = st.sidebar.button('Fit the model!')
    if init_btn and st.session_state.data_ready:
        with st.spinner("Initializing"):
            X_training = st.session_state.exp_hist[param_names]
            y_training = st.session_state.exp_hist[['Objective']]
            scaler = MinMaxScaler().fit(X_training)
            X_training = scaler.transform(X_training)

            if task_type == 'Regression model training':
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
                AL_model = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel))
                teach_model(AL_model, X_training, y_training.to_numpy().reshape(-1))
                st.session_state.model = AL_model
                y_pred, y_std = AL_model.predict(X_training, return_std=True)
                st.session_state.curr_perf = r2_score(y_training, y_pred)

            else:
                kernel = Matern(length_scale=1.0)
                regressor = GaussianProcessRegressor(kernel=kernel)
                BO_model = BayesianOptimizer(estimator=regressor)
                teach_model(BO_model, X_training, y_training.to_numpy().reshape(-1))
                st.session_state.model = BO_model
                st.session_state.curr_perf = BO_model.get_max()[1]

            st.session_state.model_init = True
            st.success('Model initialized')

    with c1:
        st.session_state.batch_size = st.slider('Batch size', 1, 100)
        get_design = st.button('Get a new batch of designs')

    if get_design:
        if st.session_state.data_ready:
            if task_type == 'Objective optimization':
                st.session_state.util_plot, st.session_state.new_batch_idx, st.session_state.new_batch = get_design_BO(
                    st.session_state.model, st.session_state.pool, st.session_state.batch_size)
            else:
                st.session_state.util_plot, st.session_state.new_batch_idx, st.session_state.new_batch = get_design_AL(
                    st.session_state.model, st.session_state.pool, st.session_state.batch_size)
            st.session_state.show_design = True
        else:
            st.error('Please check the parameters in the two files you uploaded')

    if st.session_state.show_design:
        with c2:
            st.plotly_chart(st.session_state.util_plot, use_container_width=True)
        with c3:
            st.subheader('Next batch experiment designs')
            st.dataframe(st.session_state.new_batch)
            if st.checkbox('Show indices of the new batch'):
                st.write(st.session_state.new_batch_idx)

            new_obj_str = st.text_input('Objective values for the new batch of experiments:')
            if st.button('Update evaluated and unlabeled pool'):
                new_obj_list = list(map(float, new_obj_str.split(',')))
                new_exp = pd.concat([
                    st.session_state.new_batch.iloc[:, :-1],
                    pd.DataFrame(new_obj_list, index=st.session_state.new_batch_idx, columns=['Objective'])
                ], axis=1).reset_index(drop=True)

                st.session_state.exp_hist, st.session_state.pool, st.session_state.curr_perf = update_data(
                    new_exp,
                    st.session_state.new_batch_idx,
                    st.session_state.exp_hist,
                    st.session_state.pool,
                    st.session_state.model,
                    task_type
                )

                st.session_state.new_batch = None
                st.session_state.new_batch_idx = None
                st.session_state.show_design = False
                st.session_state.no_iter += 1

    with c1:
        st.dataframe(st.session_state.exp_hist)
        if task_type == 'Objective optimization':
            c1.subheader(f'Current best objective after {st.session_state.no_iter} iterations: {st.session_state.curr_perf}')
        else:
            c1.subheader(f'Current r^2 after {st.session_state.no_iter} iterations: {round(st.session_state.curr_perf, 4)}')

        if st.button('Reset experiment'):
            for key in st.session_state.keys():
                del st.session_state[key]
            init_session_state()

main()
