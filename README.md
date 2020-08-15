
Black_Friday_data_analysis (https://datahack.analyticsvidhya.com/contest/black-friday/)

#### -- Project Status: [Completed]
https://github.com/abdullahalhoothy/Black_Friday_data_analysis/blob/master/black%20friday/Model.ipynb

# Black Friday Sales- Analysis and Prediction
One of the most important applications of machine learning in the retail sector is the prediction of the amount that a person is willing to spend at a store. Such predictions become important because they help the store owners to plan their finances, marketing strategies and inventory.
Kaggle, thus, provided a dataset of Black Friday Sales, which consists of numerous variables that describe the customer’s purchasing patterns. This problem, specifically, is a linear regression problem where the dependent variable (amount of purchase) is to be predicted with the help of information derived from other variables, called the independent variables.
The aim of this project thus is to predict the purchase behaviour of customers against different products. The project has been divided into various steps, which are described in detail further.

#### 1.	Importing Libraries
a.	Pandas and NumPy for data munging or data wrangling.
b.	MatplotLib and Seaborn for plotting visuals like graphs and distributions.
c.	SciPy for various statistical operations.
d.	Sklearn for model building

```Python
#importing important libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
```

#### 2.	Importing the Dataset
I have imported the dataset which I had stored in my drive and the path of which has been mentioned.
Kaggle has already divided the dataset into train and test data and I have thus stored them in the respective dataframes.
```Python
train = pd.read_csv(locationpath+"/train.csv")
test = pd.read_csv(locationpath+"/test.csv")

```

#### 3.	Data Exploration

Checking the Structure of Data
The train data has a total of 10 independent variables and 1 dependent variable, which is purchase and a total of 550068 entries. 
It further has a total of 5 integer variables, 2 float variables and 5 categorical variables and a huge number of null values in product categories 2 and 3.
```Python
# some information on data
train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 550068 entries, 0 to 550067
Data columns (total 12 columns):
 #   Column                      Non-Null Count   Dtype  
---  ------                      --------------   -----  
 0   User_ID                     550068 non-null  int64  
 1   Product_ID                  550068 non-null  object 
 2   Gender                      550068 non-null  object 
 3   Age                         550068 non-null  object 
 4   Occupation                  550068 non-null  int64  
 5   City_Category               550068 non-null  object 
 6   Stay_In_Current_City_Years  550068 non-null  object 
 7   Marital_Status              550068 non-null  int64  
 8   Product_Category_1          550068 non-null  int64  
 9   Product_Category_2          376430 non-null  float64
 10  Product_Category_3          166821 non-null  float64
 11  Purchase                    550068 non-null  int64  
dtypes: float64(2), int64(5), object(5)
memory usage: 50.4+ MB

```

Checking the Head of Data
We can see that there are a total of 9 categorical variables in the dataset, which are user ID, product ID, gender, occupation (coded in numerical values), city category, marital status (again coded as 0 and 1), product categories - 1,2 and 3 and age (in various ranges).
We can also see symbols like (+) in the columns age and stay_in _current_city_years, which have to be taken care of before running the algorithms.
```Python
train.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>




[Insert the screenshot of code train.head() and its output, 
Fig name: head of the data]

Checking the Total Number of Unique Values in the Train Data
It can be seen that there are a large number of categories for all the variables except gender, city category and marital status, which have 2, 3 and 2 categories respectively.
Also, after removing all the duplicate user ID’s, we can see that all the unique users have consistent information in all the entries except product category 2 and 3 which comprise of missing values.

```Python

```
[Insert screenshot of:
1.	code for checking unique values and its output, 
2.	code for getting the values of every unique entry from the following possible features and its output
3.	checking if unique users have consistent information in all of their entries and its output
Fig name: checking unique values]

## Visualization
Visualization is an important part of any project because it helps us to see the different patterns in the data with the help of graphs and understand the data in a better manner.
I have plotted bar graphs for all the variables in the dataset to understand the total frequency of data in different categories and how different categories are affecting the purchase amount.
3 graphs have been plotted for each variable based on:
a.	Count of the total entries,
b.	Unique user count for all the variables, and
c.	Grouping each unique group for summing the purchase amount using the groupby function.
```Python

```
![]()
[Insert screenshot of code for creating bar graphs and all the graphs, 
Fig Name: Visualising the data]

The observations are as discussed below:
##### i.	Gender
We can see that the frequency of males is more compared to females and they also tend to dominate the purchase. A possible explanation can be, females using credit or debit cards of their spouses for transactions.

##### ii.	Age
The age group 26-35 has maximum entries and has spent the most, followed by the age groups 36-45 and 18-25. This is because this category of population, usually considered as the youth is the working population and has more spending power and wants compared to the other age groups. They can thus be considered as potential buyers.

##### iii.	City Category
It can be seen that the frequency of people in city category B is maximum but the total number of unique entries for city category C are maximum. Also, people from city category B have spent the most as can be seen from the graph on axis 2.

##### iv.	Marital Status
Code 0 represents the category of people who are not married and 1 represents the ones who are married. From the graph, it can be seen that there is a higher proportion of unmarried people and they are also spending more compared to the ones who are married.

##### v.	Occupation
We can see that Occupation 4 has the highest frequency of people and so is their purchase amount compared to the others.

##### vi.	Stay in Current City Years
This column represents the total period of stay of a customer in the current city in years. It can be seen that people with 1 year of stay have the highest frequency and have spent the most. 

##### Outliers
Outliers are observations that lie at an abnormal distance from other values in the data. It is important to treat outliers because their presence in the data can lead to biased results.
I have plotted a boxplot for detecting the outliers in the purchase variable. We can see the outliers are present in this variable and I might discard them later if they are very far from the maximum range.
```Python

```
![]()
[Insert the code for creating boxplot and the image of the boxplot, 
Fig name: Detecting Outliers]

### 4.	Pre-Processing

##### Filling the Null Values
There are a large number of null values in the columns product category 2 and product category 3 and I have filled them with 0 to avoid:
a.	Errors while model building and 
b.	Bias by filling them with existing values.
Another reason for filling in this value was that it did not impact the accuracy of the model.
```Python

```
[Insert screenshot of the code where you have filled null values with zero in product category 2 and 3, 
Figure name: Filling the null values]

##### Data Manipulation
It is important to encode all the categorical variables for better model accuracy and I have used label encoder for encoding the following variables in both, the train and the test dataset:

a.	Gender
0 for female and 1 for male.

b.	Age
Encoded the various categories in age as follows:
0-17 years – 0
18-25 years – 1
26-35 years – 2
36-45 years – 3
46-50 years – 4
52-55 years – 5
55 + years – 6

c.	City Category
Encoded city category A as 0, category B as 1 and category C as 2.

d.	Stay in Current City Years
I have used one-hot encoding for this variable and converted it to dummies where the categories are 0 and 1.
```Python

```
[Add codes for:
1.	Turn gender into binary for training data set
2.	Giving Age groups Numerical values for training data set
3.	creating dummy variables for city category for training data set
4.	New variable for outlet
5.	Turn gender binary for test data set
6.	Giving Age Numerical values for test data set
7.	New variable for outlet
8.	Dummy Variables 
(all these are the comments that you have put before coding)
Fig Name: Encoding the data]

Correlation Matrix and Heat Map
Correlation matrices and heat maps are used to understand the correlation of different variables with the target variable. 
For the correlation matrix and heat map generated for this dataset, some variables show a positive while others show a negative correlation with the target variable, purchase but the correlation with any of the variables is not very strong.
```Python

```
![]()
[Add code for creating correlation matric and heat map and the images of both,
Figure name: correlation matrix and heat map]

##### 5.	Model Building

Model Coding
I have defined the target variable i.e., purchase and the 2 other variables, user id and product id, which are required in the final submission file.
I have then defined a function, modelfit which allows to fit the algorithm on the training data for giving predictions. 
This function also allows to perform cross-validation and then print the final root mean square error (RMSE) and cross-validation (CV) score.
After training the algorithm by building a model for the train data, the same algorithm is applied to test data and the obtained results are stored in the path that I have specified. 
You can modify your storage path as required.
```Python

```
[Add screenshot of code for model coding (whole cell), 
Figure name: model coding]

##### Verbatum Model
Here, I have created 2 dataframes from the train and test data respectively by removing the variables product id and user id because ID’s are not required in final model building as they are assumed to have no effect on the target variable and purchase from the training data because it is the target variable.
I have checked the dimensions and head of the new dataframes further.
```Python

```
[Add screenshots of all the codes from cell number 61 to 69 in your Github repository, 
![]()
Figure name: Verbatum model]

#### Linear Regression Model
Next, I have built a linear regression model using the defined function modelfit as discussed above and obtained the following results:
a.	RMSE score of 4625,
b.	CV scores as:
Mean - 4628 | Std - 33.74 | Min - 4555 | Max – 4684, which are the scores of mean value, standard deviation, minimum value and the maximum value, and 
c.	A graph which depicts the linear regression coefficients in ascending order of their values.
```Python

```
![]()
[Add screenshot of the code for building linear regression model and the model report with the graph of the model coefficients, Figure name: Linear Regression model]

#### Decision Tree Regressor Model
I have further built a decision tree model by again removing the id’s, target variable and product category 2 using the function modelfit and obtained the following results
a.	RMSE score of 2983, which has improved considerably compared to the linear regression model as tree models usually give more levels of accuracy,
b.	CV scores as:
CV Score : Mean - 2998 | Std - 20.42 | Min - 2959 | Max – 3034 
And it can be seen that the standard deviation has improved considerably compared to linear regression model
c.	A bar graph that shows feature importance and as per this graph, product category 1 has the highest impact on the purchase amount.
```Python

```
![]()
[Add screenshot of the code for building decision tree model and the model report with the graph of the feature importance, 
Figure name: Decision Tree Regressor model]

#### XGB Regressor Model
The last model that I have built is the XGB Regressor Model by importing XGBRegressor package form the library xgboost.
This model gives the mean absolute error of 220.69 and RMSE score of 2965 which is more than that of the tree model. 
```Python

```
![]()
[Add screenshot of the code for building xgb model and the model report, 
Figure name: XGB Regressor model]


## Summing Up
Machine learning can be used for performing a variety of tasks, one of them being giving predictions for the future data based on the past data. In this project, we used machine learning to predict the amount that a customer is likely to spend during the Black Friday sales. We started by exploring the data to find interesting patterns and trends in the data and finally applied various algorithms for predicting the purchase amount. 
From the models that we built, we can finally infer that decision tree regressor model has given the highest accuracy and lowest error compared to the other two models.








