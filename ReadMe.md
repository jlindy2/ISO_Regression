# ISO Regression Analysis
### By: Jeff Lindberg

## Executive Summary
The goal of this analysis find significant indicators to describe and predict a player’s power production. Using ISO as my target variable, I ran a multivariable regression model to find the most important features. These features can be used to empower the front office to make informed decisions on evaluating players for arbitration and free agent contracts. Also, to developing players in the organization to maximize value and put the best product on the field. This can even be used in drafting and scouting other leagues where the data is available.

## Contents
1. [Introduction](#introduction)
    - [Problem Statement](#problem_statement)
    - [Dataset](#dataset)
2. [Analysis](#analysis)
    - [Data Cleaning](#data_cleaning)
    - [Exploratory Analysis](#exploratory_analysis)
    - [Modeling](#modeling)

## Introduction <a name="introduction"></a>

### Problem Statement <a name="problem_statement"></a>
As a front office in an organization, it’s important to find the best players available and be able to make your players better. Can we find significant indicators to describe and predict a player’s power production? 

### Dataset <a name="dataset"></a>
The analysis is based on a dataset from Statcast (https://baseballsavant.mlb.com/) that includes observations from 2015-2019. The observations include metrics the describe the quality and direction of contact a player makes. The size of my initial dataset was (729,18).

## Analysis <a name="analysis"></a>

### Data Cleaning <a name="data_cleaning"></a>
The inital data set was very clean but I still had to drop unnecessary columns. I also used a standard scaler and then dropped the few outliers that were in there. 

### Exploratory Analysis <a name="exploratory_analysis"></a>
I found that all my features were normally distributed and most of them had a linear relationship with my target variable. There was some multicollinearity among the features that needed to be accounted for. Some of the variables did not have a significant relationship with the target amd were not used in the final model.

### Modeling <a name="modeling"></a>
My analysis used matplotlib and seaborn to create scatter plots and heatmaps to explore the data. I used statsmodels and sklearn to create my regression model and validate it.
