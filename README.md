
# Project Name
Black_Friday_data_analysis (https://datahack.analyticsvidhya.com/contest/black-friday/)


#### -- Project Status: [Completed][Polishing]
abdullahalhoothy/Black_Friday_data_analysis/Model.ipynb

## Project Intro/Objective

The purpose of this project is to build a machine learning model that predicts sale revenue and the main features which effect it most. which will help them to create personalized offer for customers against different products.  
The Features included things like age, gender, marital status, categories of products purchased.A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. 

### Methods Used
* exploratory analysis
* Descriptive analysis
* Data Visualization
* Predictive Modeling
* Linear model
* DecisionTreeRegressor
* XGboost
* Random Forests

### Technologies/packages
* Python.
* Pandas, jupyter.
* Scikit-learn.
* Seaborn.


## Project Description
Given historical purchase patterns of different products from customers, the challenge is to predict the purchase amount based on the given features.
The Features included things like age, gender, marital status, categories of products purchased.

As first step I started to understand the problem and made sure that numbers or categories meant what i thought they meant (i.e 1=married,0=single).
Then did some exploratory analysis, to find missing values and outliers there was a lot of data and i realized that it’s relatively clean with no outliers.only a some columns had missing values.
I then did some Descriptive analysis and then began with linear model and moved to DecisionTreeRegressor and currently doing XGboost and random forests. 
DecisionTreeRegressor was much better the linear models by a big margin possible because of the many different ordinal categories. 


## Needs of this project

- data exploration/descriptive statistics.
- data processing/cleaning.
- statistical modeling.
- writeup/reporting.

## Exploring the training Data
The data has 55008 enteries with the following features:(User ID, Age, City Category, Gender, Martial status, Occupation, Product ID, Purchase, Stay In Current city Years, Product Category1,2&3).
- User ID is a customer ID.
- Occupations are masked.
- Martial Status is 1=married,0=single.

![](black%20friday/IMAGES/Visualized%20data.png)


One thing that is surpising to me is that the sample is 71.2% Male buyers, however many of the market research I've read have stated that as much as 80% of consumers are women.

