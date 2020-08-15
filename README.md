
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




Checking the Total Number of Unique Values in the Train Data
It can be seen that there are a large number of categories for all the variables except gender, city category and marital status, which have 2, 3 and 2 categories respectively.
Also, after removing all the duplicate user ID’s, we can see that all the unique users have consistent information in all the entries except product category 2 and 3 which comprise of missing values.


```python
# understanding how many unique values i have
for col_name in train.columns:
    print(col_name, len(train[col_name].unique()))
```

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
    


```python
# getting the values of every unique entry from the following possible features 
for col_name in ['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years','Marital_Status']:
    print(sorted(train[col_name].unique()))
```

    ['F', 'M']
    ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ['A', 'B', 'C']
    ['0', '1', '2', '3', '4+']
    [0, 1]
    


```python
# checking if unique users have consistant information in all of their enteries 
temp = train.drop_duplicates('User_ID')
print(temp.count())
df2 = train.drop(['Purchase','Product_ID','Product_Category_1','Product_Category_2','Product_Category_3'], axis=1)
df3 = df2.drop_duplicates()
df3.count()
```

    User_ID                       5891
    Product_ID                    5891
    Gender                        5891
    Age                           5891
    Occupation                    5891
    City_Category                 5891
    Stay_In_Current_City_Years    5891
    Marital_Status                5891
    Product_Category_1            5891
    Product_Category_2            4097
    Product_Category_3            1914
    Purchase                      5891
    dtype: int64
    


## Visualization
Visualization is an important part of any project because it helps us to see the different patterns in the data with the help of graphs and understand the data in a better manner.
I have plotted bar graphs for all the variables in the dataset to understand the total frequency of data in different categories and how different categories are affecting the purchase amount.
3 graphs have been plotted for each variable based on:
a.	Count of the total entries,
b.	Unique user count for all the variables, and
c.	Grouping each unique group for summing the purchase amount using the groupby function.

```python
# lets compare the counts of entiers and unique users and the amount of purchases 
Train_unique= train.drop_duplicates('User_ID')
# Second for each unique group, we do a group by to sum the purchase amount
dfGender = train.groupby(['Gender'])['Purchase'].sum()
dfage = train.groupby(['Age'])['Purchase'].sum()
dfoccu = train.groupby(['Occupation'])['Purchase'].sum()
dfCC = train.groupby(['City_Category'])['Purchase'].sum()
dfstay = train.groupby(['Stay_In_Current_City_Years'])['Purchase'].sum()
dfM = train.groupby(['Marital_Status'])['Purchase'].sum()


fig, axes = plt.subplots(nrows=6, ncols=3, figsize=[25, 25])

# lets check how many entries do we have for each column. 
# count of entries  
train['Gender'].value_counts().plot(kind='barh', ax=axes[0,0], title='Gender')
train['Age'].value_counts().plot(kind='barh', ax=axes[1,0], title='Age')
train['City_Category'].value_counts().sort_index().plot(kind='barh', ax=axes[2,0], title='City_Category')
train['Marital_Status'].value_counts().plot(kind='barh', ax=axes[3,0], title='Marital_Status')
train['Occupation'].value_counts().sort_index().plot(kind='barh', ax=axes[4,0], title='Occupation')
train['Stay_In_Current_City_Years'].value_counts().plot(kind='barh', ax=axes[5,0], title='Stay_In_Current_City_Years')
# unique user count
Train_unique['Gender'].value_counts().plot(kind='barh', ax=axes[0,1], title='Gender')
Train_unique['Age'].value_counts().plot(kind='barh', ax=axes[1,1], title='Age')
Train_unique['City_Category'].value_counts().sort_index().plot(kind='barh', ax=axes[2,1], title='City_Category')
Train_unique['Marital_Status'].value_counts().plot(kind='barh', ax=axes[3,1], title='Marital_Status')
Train_unique['Occupation'].value_counts().sort_index().plot(kind='barh', ax=axes[4,1], title='Occupation')
Train_unique['Stay_In_Current_City_Years'].value_counts().plot(kind='barh', ax=axes[5,1], title='Stay_In_Current_City_Years')
# now lets plot amount of purchaes as per our prvious groupby
dfGender.sort_values(ascending=False).plot(kind='barh', ax=axes[0,2], title='Gender Purchase Billions$')
dfage.sort_values(ascending=False).plot(kind='barh', ax=axes[1,2], title='Age Purchase Billions$')
dfCC.sort_index().plot(kind='barh', ax=axes[2,2], title='City_Category Purchase Billions$')
dfM.sort_values(ascending=False).plot(kind='barh', ax=axes[3,2], title='Marital_Status Purchase Billions$')
dfoccu.sort_index().plot(kind='barh', ax=axes[4,2], title='Occupation Purchase Billions$')
dfstay.sort_values(ascending=False).plot(kind='barh', ax=axes[5,2], title='Stay_In_Current_City_Years Purchase Billions$')

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
```python
#finding outliers
Q1 = train['Purchase'].quantile(0.25)
Q3 = train['Purchase'].quantile(0.75)
IQR = Q3 - Q1
outliers=train[(train['Purchase'] < Q1-1.5*IQR ) | (train['Purchase'] > Q3+1.5*IQR)]['Purchase']
Outlier_index=outliers.reset_index().drop(['Purchase'], axis=1)
oil=Outlier_index["index"].tolist()
Purchase_outliers=train.loc[Outlier_index['index']]
train_nooutliers=train.drop(index=oil,axis=0)
#ploting outliers, if they are very far rom the max range then we might need to discard them 
boxplot = train.boxplot(column='Purchase')
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
```python
#because i have categorical data in many columns i need to change them to dummy binary or int 
train['Product_Category_2']=train['Product_Category_2'].fillna(0).astype("int64")
train['Product_Category_3']=train['Product_Category_3'].fillna(0).astype("int64")
test['Product_Category_2']=test['Product_Category_2'].fillna(0).astype("int64")
test['Product_Category_3']=test['Product_Category_3'].fillna(0).astype("int64")
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

```python
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#Turn gender into binary for training data set 
gender_dict = {'F':0, 'M':1}
train["Gender"] = train["Gender"].apply(lambda line: gender_dict[line])
```


```python
# Giving Age groups Numerical values for training data set 
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
train["Age"] = train["Age"].apply(lambda line: age_dict[line])
```


```python
# creating dummy variables for city category for training data set 
city_dict = {'A':0, 'B':1, 'C':2}
train["City_Category"] = train["City_Category"].apply(lambda line: city_dict[line])
```


```python
#New variable for outlet
train['Stay_In_Current_City_Years'] = le.fit_transform(train['Stay_In_Current_City_Years'])
#Hot encoding :
train = pd.get_dummies(train, columns=['Stay_In_Current_City_Years'])
train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
```


```python
#Turn gender binary for test data set
gender_dict = {'F':0, 'M':1}
test["Gender"] = test["Gender"].apply(lambda line: gender_dict[line])
test["Gender"].value_counts()
# Giving Age Numerical values for test data set
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
test["Age"] = test["Age"].apply(lambda line: age_dict[line])
test["Age"].value_counts()
city_dict = {'A':0, 'B':1, 'C':2}
test["City_Category"] = test["City_Category"].apply(lambda line: city_dict[line])
test["City_Category"].value_counts()
le = LabelEncoder()
```


Correlation Matrix and Heat Map
Correlation matrices and heat maps are used to understand the correlation of different variables with the target variable. 
For the correlation matrix and heat map generated for this dataset, some variables show a positive while others show a negative correlation with the target variable, purchase but the correlation with any of the variables is not very strong.

```python
# calculate the correlation matrix
corr = train.corr()

# display the correlation matrix
display(corr)

# plot the correlation heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')
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
```python
#Define target and ID columns:
target = 'Purchase'
IDcol = ['User_ID','Product_ID']
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors],(dtrain[target]) , cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    #submission.to_csv(
    #    r"C:/Users/u_je_/GoogleDrive/Personal/Work/Online/Jupyter/Git/Black friday Analysis Github repository/Black_Friday_data_analysis/black friday/output/test1.csv",index=False)
```
##### Verbatum Model
Here, I have created 2 dataframes from the train and test data respectively by removing the variables product id and user id because ID’s are not required in final model building as they are assumed to have no effect on the target variable and purchase from the training data because it is the target variable.
I have checked the dimensions and head of the new dataframes further.


```python
model =LinearRegression()
predictorsdata=train.drop(['Purchase','Product_ID','User_ID'],axis=1)
try:
    testdata=test.drop(['Purchase','Product_ID','User_ID'],axis=1)
except:
    testdata=test.drop(['Product_ID','User_ID'],axis=1)
model.fit(predictorsdata,train['Purchase'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
predictorsdata.shape
```




    (550068, 12)




```python
testdata.shape
```




    (233599, 12)




```python
model.coef_
```




    array([ 475.84555737,  106.25046973,    5.75303662,  316.21680564,
            -48.74916127, -348.07352369,   12.53884509,  143.67043618,
            -51.65926997,  -17.58148817,    9.94148796,  -21.93961919])




```python
model.intercept_
```




    9543.96177779093




```python
predictiontest=model.predict(testdata)
```


```python

```


```python
test.shape
```




    (233599, 14)




```python
test.insert(11,'Purchase',predictiontest)
```


```python
test
```


#### Linear Regression Model
Next, I have built a linear regression model using the defined function modelfit as discussed above and obtained the following results:
a.	RMSE score of 4625,
b.	CV scores as:
Mean - 4628 | Std - 33.74 | Min - 4555 | Max – 4684, which are the scores of mean value, standard deviation, minimum value and the maximum value, and 
c.	A graph which depicts the linear regression coefficients in ascending order of their values.

```python
# running LinearRegression
from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize=True)

predictors = train.columns.drop(['Purchase','Product_ID','User_ID'])
modelfit(LR, train, test, predictors, target, IDcol, 'LR.csv')

coef1 = pd.Series(LR.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
```

    
    Model Report
    RMSE : 4625
    CV Score : Mean - 4628 | Std - 33.74 | Min - 4555 | Max - 4684
    
![]()
[Add screenshot of the code for building linear regression model and the model report with the graph of the model coefficients, Figure name: Linear Regression model]

#### Decision Tree Regressor Model
I have further built a decision tree model by again removing the id’s, target variable and product category 2 using the function modelfit and obtained the following results
a.	RMSE score of 2983, which has improved considerably compared to the linear regression model as tree models usually give more levels of accuracy,
b.	CV scores as:
CV Score : Mean - 2998 | Std - 20.42 | Min - 2959 | Max – 3034 
And it can be seen that the standard deviation has improved considerably compared to linear regression model
c.	A bar graph that shows feature importance and as per this graph, product category 1 has the highest impact on the purchase amount.
```python
# running DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor
RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
predictors = train.columns.drop(['Purchase','Product_ID','User_ID','Product_Category_2'])
modelfit(RF, train, test, predictors, target, IDcol, 'RF.csv')

coef4 = pd.Series(RF.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')
```

    
    Model Report
    RMSE : 2983
    CV Score : Mean - 2998 | Std - 20.42 | Min - 2959 | Max - 3034
    
![]()
[Add screenshot of the code for building decision tree model and the model report with the graph of the feature importance, 
Figure name: Decision Tree Regressor model]

#### XGB Regressor Model
The last model that I have built is the XGB Regressor Model by importing XGBRegressor package form the library xgboost.
This model gives the mean absolute error of 220.69 and RMSE score of 2965 which is more than that of the tree model. 
```python
#installing xhboost model 
!pip install xgboost
```

    Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (0.90)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.18.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.4.1)
    


```python
# building XGBRegressor model 
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train[predictors], train[target], early_stopping_rounds=5, 
             eval_set=[(train[predictors], train[target])], verbose=False)
#Predict training set:
train_predictions = my_model.predict(train[predictors])

# make predictions
predictions = my_model.predict(test[predictors])

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train[target]).values, train_predictions)))

```

    [16:01:02] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    Mean Absolute Error : 220.69855398121658
    RMSE : 2965
![]()
[Add screenshot of the code for building xgb model and the model report, 
Figure name: XGB Regressor model]


## Summing Up
Machine learning can be used for performing a variety of tasks, one of them being giving predictions for the future data based on the past data. In this project, we used machine learning to predict the amount that a customer is likely to spend during the Black Friday sales. We started by exploring the data to find interesting patterns and trends in the data and finally applied various algorithms for predicting the purchase amount. 
From the models that we built, we can finally infer that decision tree regressor model has given the highest accuracy and lowest error compared to the other two models.








