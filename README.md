# Black_Friday_data_analysis

#Problem Statement.

Given historical purchase patterns of different products from customers, the challenge was to predict the purchase amount based on the given features.
The Features included things like age, gender, marital status, categories of products purchased.

#Approach.

As first step i started to analyze the problem and made sure that numbers or categories meant what i thought they meant (i.e 1=married,0=single)
Then did some exploratory analysis, to find missing values and outliers there was a lot of data and i realized that it’s relatively clean with no outliers.only a some columns had missing values.
I then did some Descriptive analysis and then began with linear model and moved to DecisionTreeRegressor and currently doing XGboost and random forests. 
DecisionTreeRegressor was much better the linear models by a big margin possible because of the many different ordinal categories 
