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

In Bayesian Optimization, Gaussian Process is the commonly used surrogate model to model the objective function by incorporating knowledge of the previously evaluated data instances. While selecting the next experiment(s) to evaluate, to balance the probability of exploiting near the current optimal design and exploring in the region with the greatest uncertainty, utility functions are often applied. Upper Confidence Bound (UCB) is one of the most commonly used utility functions, which primarily favors exploration, but switches to favoring exploitation around the current optimal design as gaining more information elsewhere it has explored.

*  **Active learning**

In scientific research, optimization of measurements by tuning parameters and regression model training are two common problems. In many experiments, there are multiple design parameters (or independent variables) that can take multiple or even infinite numbers of values, which makes the searching space very large and hence an exhaustive search infeasible or impossible. 

Thus, Bayesian Optimization, which utilizes the historical data as prior knowledge, fits a probabilistic surrogate function and suggests a new candidate likely to improve the modeling of the objective function for evaluation according to the posterior, could help researchers find optimal designs faster.

In Bayesian Optimization, Gaussian Process is the commonly used surrogate model to model the objective function by incorporating knowledge of the previously evaluated data instances. While selecting the next experiment(s) to evaluate, to balance the probability of exploiting near the current optimal design and exploring in the region with the greatest uncertainty, utility functions are often applied. Upper Confidence Bound (UCB) is one of the most commonly used utility functions, which primarily favors exploration, but switches to favoring exploitation around the current optimal design as gaining more information elsewhere it has explored.


## Related Work
* **Data**
1.	Protein Directed Evolution
In the study published by Wu et al., , the researchers aimed to find the mutant of the protein GB1 with the highest “fitness” (a measurement of the protein’s stability and functionality) by mutating the amino acid at four positions in its protein sequence. There are 20 possible amino acids at each locus in theory, but because of experimental constraints, the researchers were not able to produce all 204 variants, but only 149,361 of them. The fitness scores for the mutant are relative to the original wild-type protein, meaning only those with fitness scores> 1 are considered beneficial. And it was found that only 2.4% of the mutants are beneficial.

2.	Buffer Composition 
A cell-free system is an in vitro system allowing researchers to examine biological processes. While performing the experiments using this tool, the buffer composition can cause variation in the production efficiency of the system. In the study by Borkowski et al., they aimed to optimize the yield of a specific protein by tuning the concentration of 11 different substances in the buffer and to train a regression model in an active learning fashion to predict the yield of specific buffer compositions.

## Methods

## Results

## Discussion

## Future Work
