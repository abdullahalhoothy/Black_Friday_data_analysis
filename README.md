
# Project Name
Black_Friday_data_analysis (https://datahack.analyticsvidhya.com/contest/black-friday/)


#### -- Project Status: [Completed][Polishing]
https://github.com/abdullahalhoothy/Black_Friday_data_analysis/blob/master/black%20friday/Model.ipynb

## Project Intro/Objective
A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. 
The purpose of this project is to build a machine learning model that predicts sale revenue and the features which effect it most. This will help the company create personalized offers for customers for products that the customer may like.  
The Features included: age, gender, marital status, categories of products purchased.


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

Data columns (total 12 columns):
User_ID                       550068 non-null int64
Product_ID                    550068 non-null object
Gender                        550068 non-null object
Age                           550068 non-null object
Occupation                    550068 non-null int64
City_Category                 550068 non-null object
Stay_In_Current_City_Years    550068 non-null object
Marital_Status                550068 non-null int64
Product_Category_1            550068 non-null int64
Product_Category_2            376430 non-null float64
Product_Category_3            166821 non-null float64
Purchase                      550068 non-null int64

![](black%20friday/IMAGES/Visualized%20data.png)


One thing that is surpising to me is that the sample is 71.2% Male buyers, however many of the market research I've read have stated that as much as 80% of consumers are women.


# understanding how many unique values i have
for col_name in train.columns:
    print(col_name, len(train[col_name].unique()))
User_ID 5891
Product_ID 3631
Gender 2
Age 7
Occupation 21
City_Category 3
Stay_In_Current_City_Years 5
Marital_Status 2
Product_Category_1 20
Product_Category_2 18
Product_Category_3 16
Purchase 18105


# getting the values of every unique entry from the following possible features 
for col_name in ['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years','Marital_Status']:
    print(sorted(train[col_name].unique()))
['F', 'M']
['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
['A', 'B', 'C']
['0', '1', '2', '3', '4+']
[0, 1]

