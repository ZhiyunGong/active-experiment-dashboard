# Final Project Proposal

**GitHub Repo URL**: TODO

**Track:** Application

## A web application for interactive Active Learning and Bayesian Optimization
### Introduction
Different from conventional machine learning algorithms, which need to be trained on a large amount of labeled data collected beforehand, Active Learning starts with a small number of data points, and interactively suggest new instances to evaluate, and aims to maximize the performance metrics of the model using as few as possible data points.

Bayesian Optimization (BO) is an optimization method that utilizes historical data as prior knowledge, fits a probabilistic surrogate function, and suggests new candidates with the highest utility values to evaluate according to the posterior of the probabilistic model.

In scientific research, there are usually experimental parameters, hardware/software configurations, and material compositions that need to be optimized. However, In many cases, the actual cost in terms of money or time makes it very inefficient or even impossible to evaluate all possible combinations of parameters to find the optimal solution. An application allowing the users to get suggestions for experiment parameters for training a good regression model for the objective value or optimizing the objective value would be helpful for researchers who are not familiar with performing active learning or BO in a computing environment.

### Data
This dataset was published with the research paper “Adaptation in protein fitness landscapes is facilitated by indirect paths” (available at: https://elifesciences.org/articles/16965 ). The main optimization problem in the study is to modify 4 loci (20 possible amino acids at each locus, 204 combinations in total) in the sequence of GB1 protein, in order to find the variant with the highest “fitness”, which is a measure of the protein’s stability and function. 

The application will be tested using this dataset to explore how many evaluations will be needed to train a regression model using active learning with reasonably high accuracy on the rest of the data points, as well as how many evaluations needed for the BO algorithm to find an optimal design of the protein.

### Objectives 
This project will be on the application track. 

The application will be able to perform two tasks: 1) to train a regression model using active learning with high accuracy in predicting the objective values; 2) to find the best objective value with significantly fewer evaluations using Bayesian Optimization. 

The user interface will allow the users to specify optimization parameters as well as the legit values each parameter could take (assuming all parameters are discrete as in the dataset I plan to use, continuous variables need to be discretized), upload previous evaluations if available, task (regression model training or objective optimization), query strategy, as well as the number of experiments they want to evaluate in the next iteration. The user can then enter the objective values for the new experiments, get new performance metrics of the model, and decide whether to proceed with additional iterations. Once the user is satisfied with the current accuracy of the regression model or the current best objective, they may stop and download the full history of experiments, as well as the trained model in the model training mode.
