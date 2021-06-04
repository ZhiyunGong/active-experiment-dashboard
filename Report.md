# Final Project Report

**Project URL**: https://share.streamlit.io/cmu-ids-2021/fp--zhiyun/main/main.py

Active Learning is a special case in Machine Learning that starts with a small number of data points, and interactively suggest new instances to evaluate, and aims to maximize the performance metrics of the model using as few as possible data points. Bayesian Optimization is an efficient way to optimize a black-box function faster by suggesting new parameters based on the prior knowledge of evaluated data. These two techniques can help with training a regression model on or maximize/minimize a black-box function in different areas of scientific research. The application built in the project is a dashboard where researchers who are not familiar with running Bayesian Optimization or Active Learning experiments programmatically can complete the tasks in an interactive and user-friendly way.

## Introduction
*  **Gaussian Process**

![Gaussian Process](https://planspace.org/20181226-gaussian_processes_are_not_so_fancy/img/predictive_mean_and_range.png)

Gaussian Process regression is a commonly used model to approximate black-box functions based on prior knowledge and to make predictions in the unknown area with uncertainty.
*  **Bayesian Optimization**

In scientific research, optimization of measurements by tuning parameters and regression model training are two common problems. In many experiments, there are multiple design parameters (or independent variables) that can take multiple or even infinite numbers of values, which makes the searching space very large and hence an exhaustive search infeasible or impossible. 

Thus, Bayesian Optimization, which utilizes the historical data as prior knowledge, fits a probabilistic surrogate function and suggests a new candidate likely to improve the modeling of the objective function for evaluation according to the posterior, could help researchers find optimal designs faster.

In Bayesian Optimization, Gaussian Process is the commonly used surrogate model to model the objective function by incorporating knowledge of the previously evaluated data instances. While selecting the next experiment(s) to evaluate, to balance the probability of exploiting near the current optimal design and exploring in the region with the greatest uncertainty, utility functions are often applied. Upper Confidence Bound (UCB) is one of the most commonly used utility functions, which primarily favors exploration, but switches to favoring exploitation around the current optimal design as gaining more information elsewhere it has explored. In each iteration, the experimental design(s) with the highest utility will be selected to evaluate next.

*  **Active learning**

Active learning aims to train machine learning models more efficiently. Instead of requiring all training data to be labeled beforehand, active learning starts with a small training set, and interactively queries labels for more instances in the region of the design space where the model is most uncertain. When training a regression model in an active learning way, the model should be able to estimate the uncertainty of the unlabeled data and to model the unknown function with decent variability and less bias (eg. a simple linear regression model may have too much bias and fail to model the complexity of the black-box function). Thus, a Gaussian Process regression model seems to be a very good fit in this scenario.

## Related Work
* **Data**
1.	Protein Directed Evolution
In the study published by Wu et al., , the researchers aimed to find the mutant of the protein GB1 with the highest “fitness” (a measurement of the protein’s stability and functionality) by mutating the amino acid at four positions in its protein sequence. There are 20 possible amino acids at each locus in theory, but because of experimental constraints, the researchers were not able to produce all 204 variants, but only 149,361 of them. The fitness scores for the mutant are relative to the original wild-type protein, meaning only those with fitness scores> 1 are considered beneficial. And it was found that only 2.4% of the mutants are beneficial.

2.	Buffer Composition 
A cell-free system is an in vitro system allowing researchers to examine biological processes. While performing the experiments using this tool, the buffer composition can cause variation in the production efficiency of the system. In the study by Borkowski et al., they aimed to optimize the yield of a specific protein by tuning the concentration of 11 different substances in the buffer and to train a regression model in an active learning fashion to predict the yield of specific buffer compositions.

## Methods
The models for both tasks are implemented by the modAL and sci-kit learn packages. There are no criteria for the performance of the model to be considered as good, because the expectation may vary drastically in different fields of research and types of experiment. It depends on the users’ field-specific knowledge to decide when to stop the iterative optimization process.

* **Active Regression**

The regression model used in this app is a Gaussian Process (GP) model with a Radial Basis Function kernel. The criteria used to select non-evaluated designs for evaluation is the uncertainty of the predictions (standard deviation at those points in the Gaussian Distribution) made by the GP model trained on previously known data. In each iteration, new designs of the batch size specified by the user will be given by the application for the user to run further experiments accordingly and provide labels/measurements for them later. The performance measurement of the model is the r<sup>2</sup> on its training data, and it’s updated after each time newly evaluated instances are incorporated.


* **Bayesian Optimization**


The surrogate model is a GP model with a Matern kernel. The Upper Confidence Bound utility function is applied, and instances of the batch size specified by the user with the highest UCB scores are returned for further evaluation. The performance measurement in this case is simply the highest objective value seen so far in the evaluated experiments.

## Results
* **User Interface Design**
The user interface consists of two parts: the sidebar for settings, and the main body for performing the iterative model training or optimization. 

In the sidebar, the user will first select whether they want to perform “Regression model training” or “Objective optimization”. Then the text files containing evaluated experiments and a pool of unlabeled instances need to be uploaded. After uploading both files and clicking on the “Fit the model” button, the user can move to the main body to start the training or optimization.

In the main body, there are three columns: **1) Left:** batch size selection, all labeled data that has been learned by the model, current performance (r<sup>2</sup> or maximum objective); **2) Middle:** a scatter plot of all data instances in the unlabeled pool projected on the first 2 principal components, colored according to their utility (for Bayesian Optimization) or uncertainty (for Active Regression) level; **3) Right:** a new batch of parameter designs to be evaluated, text input for the objective values of the new experiments, an update button.

![User interface](https://github.com/CMU-IDS-2021/fp--zhiyun/blob/main/imgs/app_regression.png)  
## Discussion

## Future Work
