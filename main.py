# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:58:10 2021

@author: Zhiyun Gong
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn import preprocessing

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model = st.beta_container()

with header:
    st.title('Welcome to my project!')
    st.text('In the project, I investigated whether supervised Machine Learning models trained on screening results can be predictive of the true disgnosis')
    
    
with dataset:
    st.header('Autism screening dataset')
    st.text('Links to the datasets')
    
    df_adult = pd.read_csv('Autism_Data.arff')
    st.write(df_adult.head())
    
    st.header('Gender distribution of ASD cases in the dataset')
    gender_labels = ['Male','Female']
    values = df_adult[df_adult['Class/ASD']=='YES']['gender'].value_counts().tolist()
    fig1 = go.Figure(data = [go.Pie(labels = gender_labels, values = values)])
    st.plotly_chart(fig1)
    
    st.header('Age distribution of ASD')
    age_df = df_adult[['age','Class/ASD']]
    
    fig2 = px.histogram(age_df, x = 'age')
    st.plotly_chart(fig2)
    

    
    
    
    age_counts = []
    ages = age_df.age.unique().tolist()
    ages.remove('?')
    for a in ages:
        count = age_df.loc[age_df['age'] == a].groupby('Class/ASD').count()
        if len(count) == 2:
            age_counts.append([a,count.iloc[0,0], 'No'])
            age_counts.append([a,count.iloc[1,0], 'Yes'])

            
        elif 'YES' in count.index.to_list():
            age_counts.append([a,count.iloc[0,0], 'Yes'])
        else:
            age_counts.append([a, count.iloc[0,0], 'No'])
    age_counts_df = pd.DataFrame(age_counts, columns=['age','count','diagnosis']).sort_values(by = 'age', axis=0)
    st.write(age_counts_df)
    
    fig3 = alt.Chart(age_counts_df).mark_bar().encode(
        y = alt.X('sum(count)',stack = "normalize"),
        x = 'age',
        color = 'diagnosis'
        )
    
    st.altair_chart(fig3)
    
    # ----------------- Ethnic distribution -----------

    ethnic_counts = []
    ethnic_groups =  df_adult.loc[df_adult['Class/ASD']=='YES',['Class/ASD','ethnicity']].groupby('ethnicity').count()
    
    ethnic_labels = df_adult.ethnicity.unique().tolist()
    eth_values = df_adult[df_adult['Class/ASD']=='YES']['ethnicity'].value_counts().tolist()
    fig4 = go.Figure(data = [go.Pie(labels = ethnic_labels, values = eth_values)])
    st.plotly_chart(fig4)
        
    
    
 
    
with features:
    
    # Binarize labels
    lb = preprocessing.LabelBinarizer()
    labels = df_adult['Class/ASD']

    y = lb.fit_transform(labels)
    st.header('Features for prediction')
    st.subheader('Screening results only')
    # only contains answers to 10 questions
    X_1_1 = df_adult.iloc[:,:10]
    
    st.text("Screening questions:")
    question_df = pd.DataFrame({
        'Id': [ 'Question ' + str(i+1) for i in range(10)],
        'Question': ['I often notice small sounds when others do not',
                     'I usually concentrate more on the whole picture, rather than the small details',
                     'I find it easy to do more than one thing at once',
                     'If there is an interruption, I can switch back to what I was doing very quickly',
                     'I find it easy to ‘read between the lines’ when someone is talking to me',
                     'I know how to tell if someone listening to me is getting bored ',
                     'When I’m reading a story I find it difficult to work out the characters’ intentions ',
                     'I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc)',
                     'I find it easy to work out what someone is thinking or feeling just by looking at their face',
                     'I find it difficult to work out people’s intentions']
        })
    st.table(question_df)
    
    st.header('Projection using t-SNE')
    tsne = TSNE(n_components = 2, random_state = 0)
    projections = tsne.fit_transform(X_1_1)
    
    fig4 = px.scatter(
        projections, x=0, y = 1,
        color = labels, labels = {'color':'Diagnosis'})
    st.plotly_chart(fig4)
    

with model:
    st.header('Model training!')
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X_1_1, y, test_size = 0.1, random_state =42)
    
    #------------------- Logistic Regression ---------------
    st.header('Logistic Regression')
    
    lr1 = LogisticRegression(random_state=4)
    st.write(cross_val_score(lr1, X_1_1, y, cv = 10))
    
    lr1.fit(X_train_1, y_train_1)
    y_score_lr1_1 = lr1.predict_proba(X_test_1)[:,1]
    
    fpr_lr1_1, tpr_lr1_1, thresholds = roc_curve(y_test_1, y_score_lr1_1)
    fig5 = px.area(
        x = fpr_lr1_1, y= tpr_lr1_1,
        title = f'ROC curve (AUC = {auc( fpr_lr1_1,tpr_lr1_1)}')
    st.plotly_chart(fig5)
    
    #------------ SVM  ---------------------
    
    st.header('Support Vector Machine')
    svm1 = LinearSVC(random_state=0)
    st.write(cross_val_score(svm1, X_1_1, y, cv = 10))
    
    #------------ Random Forest ---------------------
    
    st.header('Random Forest')
    rf1 = RandomForestClassifier(random_state=0)
    st.write(cross_val_score(rf1, X_1_1, y, cv=10))
    
    rf1.fit(X_train_1, y_train_1)
    y_score_rf1_1 = rf1.predict_proba(X_test_1)[:,1]

    
    
    fpr_rf1_1, tpr_rf1_1, thresholds = roc_curve(y_test_1, y_score_rf1_1)
    fig6 = px.area(
        x = fpr_rf1_1, y= tpr_rf1_1,
        title = f'ROC curve (AUC = {auc( fpr_rf1_1,tpr_rf1_1)}:.3f)')
    st.plotly_chart(fig6)
    
    
    #------------- 
