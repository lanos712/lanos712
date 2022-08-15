#!/usr/bin/env python
# coding: utf-8

# # BIKER ASSIGNMENT | Machine Learning | Regression Analysis

# ### Steps to perform
# - Loading and Understanding the Data 
# - Preparing data for modeling
# - Train the model
# - Residual Analysis
# - Prediction and evaluation on test set

# # Section I - Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score


# In[4]:


import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## Section II - Loading and Understanding the Dataset 

# ### In Section 2 
# - Load the Data set
# - Understand the variables present in Dataset
# - Transform the dataset 
# 

# ### Load the Data Set

# In[5]:


df = pd.read_csv('day.csv')


# In[ ]:





# In[6]:


df.head()


# ### Understand the Data set 

# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


len(df[df.duplicated()])


# In[10]:


df.shape


# ### Data Inferences 
# 
# - The given data set contains 730 rows and 16 columns
# - There exists no null values and duplicate values
# - The given data set has data types as int64, object, float64.
# - 11680 is the size of data 

# ##### Points to Note about the data set from data dictionary 
# 
# - instant : It has index associated with it, all the values are unique.
# - dteday : It is a date, present in "dd/mm/yyyy" format
# - season : season (1:spring, 2:summer, 3:fall, 4:winter)
# - the column 'yr' with two values 0 and 1 indicating the years 2018 and 2019 respectively
# - mnth : mnth has values from 1-12, which will be converted into Jan-Dec
# - holiday : its a binary variable which depict 1- if there's holiday, 2- if there's not a holiday
# - weekday : the day of the week, has values from 0-7
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# - 'weathersit' and 'season' have values as 1, 2, 3, 4 which have specific labels associated with them 
#    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#    - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp : temp in Celsius
# - atemp : feeling temp in Celsius
# - hum : humidity 
# - windspeed : windspeed 
# - 'casual' indicates the number casual users who have made a rental. 
# - 'registered' on the other hand shows the total number of registered users who have made a booking on a given day
# - cnt' variable indicates the total number of bike rentals, including both casual and registered.
# - The model should be built taking this 'cnt' as the target variable.

# ### Transform the DataSet

# ###### Based upon the data dictionaries 
# ######  let's transform "season", "weathersit", "mnth"  and "weekdays" as per the data dictionary
# 

# In[11]:


#transforming season 

df['season'] = df['season'].map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})

#transform weathersit

df['weathersit'] = df['weathersit'].map ({1:"Clear to Partly cloudy", 2: "Mist and Cloudy", 
                                       3: "Light Rain or Snow", 4:"Heavy Rain or Snow"})

#transform mnth

df['mnth' ]= df['mnth'].map({1:"Jan", 2:"Feb", 3:"Mar", 4: "Apr",
                            5: "May", 6: "Jun", 7:"Jul", 8: "Aug",
                            9: "Sept" , 10 : "Oct", 11 : "Nov", 12 : "Dec"})

# transform weekdays 
df['weekday'] = df['weekday'].map({0: 'Sun', 1: 'Mon', 2 :'Tue',
                                     3: 'Wed', 4:'Thurs', 5: 'Fri',
                                     6: 'Sat', 7: 'Sun'})


# In[12]:


df.head()


# #### Tranforming Data type
# 
# -dteday has an object type but it is date
# -Converting casual,registered,cnt as float

# In[13]:


df.info()


# In[14]:


df.info()


# In[15]:


v_list = ["casual","registered","cnt"]

for var in v_list: 
    df[var] = df[var].astype("float")


# In[16]:


df.dteday.astype('datetime64')


# # Section III - Exploratory Data Analysis

# ### In Section III
# - Univariate Analysis
# - Bivariate Analysis
# - Correlation
# - Dropping the irrelevant, unrequired columns 
# 

# ### Univariate Analysis

# In[17]:


#Analysing Target Variable

sns.color_palette("Paired")


# subplot grid:
fig, ax=plt.subplots(nrows =1, ncols=2, figsize= (12,5))

#main title
fig.suptitle('Analysis of Target Variable (CNT)', fontsize=20)

#plotting plot 1
sns.boxplot(y=df['cnt'], ax= ax[0], color="skyblue")
ax[0].set_title("boxplot of Target Variable (CNT)", fontsize=12, fontweight=20, y= 1.02)

ax[0].set_ylabel(" ")
ax[0].set_xlabel(" ")

#plotting plot 2 
sns.distplot(df['cnt'],ax= ax[1])
ax[1].set_title("distplot of Target Variable (CNT)", fontsize=12, fontweight=20, y= 1.02)

ax[1].set_ylabel(" ")
ax[1].set_xlabel(" ")

plt.tight_layout()
plt.show()


# #### Data Inferences 
# 
# - There exist no outliers in the target variable.
# - The distribution is normal

# ### Analysing Numerical Variables

# In[18]:


## Let's analysis continuous independent variables 

#defining a list for the variables

cont_var = [i for i in df.select_dtypes(exclude='object').columns 
            if df[i].nunique() >2 and i !='cnt''dteday']


# In[19]:


cont_var


# In[20]:


#Analysing Continuous Variable


# subplot grid:
fig, ax=plt.subplots(nrows =1, ncols=len(cont_var), figsize= (12,5))

#main title
plt.suptitle('Analysis of Continuous Variable', fontsize=20)

#plotting plot 
for i in range(len(cont_var)):
    
    sns.boxplot(y= df[cont_var[i]], ax= ax[i])
    
    ax[i].set_title(f'{cont_var[i].title()}', fontsize=15)
    ax[i].set_xlabel(" ")
    ax[i].set_ylabel(" ")

plt.tight_layout()
plt.show()


# In[21]:


#Analysing Continuous Variable


# subplot grid:
fig, ax=plt.subplots(nrows =1, ncols=len(cont_var), figsize= (25,5))

#main title
fig.suptitle('Analysis of Continuous Variable', fontsize=20)

#plotting plot 
for i in range(len(cont_var)):
    
    sns.distplot(df[cont_var[i]], ax= ax[i])
    
    ax[i].set_title(f'{cont_var[i].title()}', fontsize=20)
    ax[i].set_ylabel(" ")
    ax[i].set_xlabel(" ")

plt.tight_layout()
plt.show()


# #### Data Inferences 
# 
# - Humiditity and windspeed are rightly skewed
# - Registered and the target variable have a similar distribution
# - Windspeed is slightly left skewed.
# 

# In[22]:


df.info()


# ### Analysing Categorical Variable

# In[23]:


## Let's analysis continuous independent variables 

#defining a list for the variables

catg_var = [i for i in df.select_dtypes(include='object', exclude='datetime64').columns]
catg_var.extend(i for i in df.columns if df[i].nunique()==2)


# In[24]:



# subplot grid:
fig, ax= plt.subplots(nrows =2, ncols=int(len(catg_var)/2), figsize= (22,8))

#main title
                      
fig.suptitle('Analysis of Categorical Variables', fontsize=20, fontweight =20, y=0.99)

#plotting plot 
k=0
for i in range(2): 
        for j in range (int(len(catg_var)/2)):
        
            sns.countplot(x=df[catg_var[k]], ax= ax[i, j], palette = "vlag")
    
            ax[i, j].set_title(f'{catg_var[k].title()}', fontsize=20)
            ax[i, j].set_ylabel(" ")
            ax[i, j].set_xlabel(" ")
            
            k+=1


plt.tight_layout()
plt.show()


# #### Data Inferences 
# 
# - Dteday column should be dropped
# - Fall is the longest season out of all the 4 seasons
# - The weather situation is mostly "Clear to Partly Cloud".
# 

# ### Bivariate Analysis

# In[25]:


# Anaylysing Categorical data against target variable 'cnt' 
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x= "mnth", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,2)
sns.boxplot(x= "weekday", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,3)
sns.boxplot(x= "weathersit", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,4)
sns.boxplot(x= "season", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,5)
sns.boxplot(x= "yr", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,6)
sns.boxplot(x= "workingday", y="cnt", data=df, palette = "vlag")
plt.subplot(3,3,7)
sns.boxplot(x= "holiday", y="cnt", data=df, palette = "vlag")
plt.show()


# #### Data Inference 
# - There's a rise in bike selling in September month
# - Fall season has a higher demand of bikes
# - On holidays, the demand is high
# - when the weather is partly cloud or clear, A weathersit has a high demand of bikes.
# - Fridays have a high demand of bikes
# 

# In[26]:


# Anaylysing Continuos variables against target variable 'cnt'


# subplot grid:
fig, ax= plt.subplots(nrows =2, ncols=int(len(cont_var)/2), figsize= (20,9))

#main title
plt.suptitle('Analysis of Continuous Variable against Target Variable', fontsize=20, fontweight = 20, y=.99)


#plotting plot 
k= 0
for i in range(2):
    for j in range(int(len(cont_var)/2)):
        sns.regplot(x = df[cont_var[k]],y= df["cnt"], ax= ax[i, j],
                   scatter_kws = {'color' : "green"} ,line_kws = {'color': "black"})
        
        ax[i, j].set_title(f'{cont_var[k].title()}', fontsize=20)
        ax[i, j].set_ylabel(" ")
        ax[i, j].set_xlabel(" ")
        
        k+=1

plt.tight_layout()
plt.show()


# #### Data Inference 
# 
# - We observe no correlation between instant and bike demand, hene we will drop this column.
# - We observe a negative correlation between windspee, humidity with bike demand 
# - We observe a similar pattern between temp and atemp, hence we will drop atemp columns.

# ### MultiVariate Analysis

# In[27]:


#Let's understand seasonwise and monthwise variation of continuous variables
df.groupby(by=["season", 'mnth']).mean()


# In[28]:


#Let's read the monthwise pattern in different weather situations
plt.figure(figsize=(20,8))

sns.countplot("weathersit", data= df, hue='mnth', palette ='Spectral')
plt.title("Monthwise Pattern in different weather situation", fontsize = 18)
plt.xlabel("Weather Type".title(), fontsize= 12)
plt.show()


# #### Data Inferences
# 
# - Maximum days are covered under Clear to Partly Cloud weather conditions
# - July has shown maximum number days as clear to partly cloud
# - The count of days are very low in light rain or snow 

# In[29]:


#Let's place a pairplot 

sns.pairplot(df, kind = "reg")
plt.show()


# #### Data Inferences
# - 'cnt' is directly  proportional to temp, atemp, casual and regitered 
# - 'cnt' is inversely proportional to windspeed and hum
# 

# ### Correlation

# In[30]:


#Let's define a list for numeric data for heatmap

df_numeric = df.select_dtypes(include=["float64"])
df_numeric


# In[31]:


# Correlation 
cor = df_numeric.corr() 

#Let's Visualise the numeric data 
plt.figure(figsize=(25,12))

#Draw Heatmap of correlation

sns.heatmap(cor,annot=True, cmap='icefire' )
plt.show()


# #### Data Inferences
# 
# - temp and atemp are highly correlated with correlation of 0.99
# - Bike demand is highly correlated with registered
# - Bike demand is negatively correlated with windspeed 
# - Humidity and bike demand are negatively correlated 

# ### Dropping the irrelevant, unrequired columns 
# 

# - From bivariate analysis, we observed 'instant','dteday' doesn't contribute to analysis and have no effect on target variable
# - Since our target variable 'cnt' is cumulative of 'casual','registered', we will drop these columns as well
# - Through heatmap, we observed temp and atemp shares a correlation of 0.99.

# In[32]:


df.shape


# In[33]:


# Let's drop atemp

df.drop("atemp", axis=1,inplace=True)
df.head(5)


# In[34]:


#Let's drop casual and registered

df.drop(['casual','registered'], axis=1, inplace = True)
df.head(5)


# In[35]:


# Let's drop dteday

df.drop(['dteday'], axis=1, inplace=True)
df.head(5)


# In[36]:


# Let's drop instant


df.drop(['instant'], axis=1, inplace= True)
df.head(5)


# In[37]:


df.shape


# In[38]:


#Let's plot heatmap again after dropping columns 



plt.figure(figsize=(10,5))

sns.heatmap(df.corr(),annot=True, cmap='icefire' )
plt.show()


# In[39]:


#Let's Analyse the growth comparision between both the years, 2018 and 2019

#creating pivot
growth_df = df.pivot_table(index = 'mnth', columns='yr', values ='cnt', aggfunc = 'mean')

#nomenclature of columns
growth_df.columns = ["2018", '2019']

# % Change
growth_df['% change'] = round(((growth_df['2019']- growth_df['2018'])/growth_df['2018'])*100, 2)


growth_df = growth_df.sort_values('% change', ascending = False)
growth_df


# In[40]:


#Let's plot the growth comparision

plt.figure(figsize=(12,7))

sns.barplot(x="mnth", y='cnt', data = df, hue = 'yr', palette="Spectral")
plt.title('Monthwise Growth over Last Year', fontsize = 20)

plt.show()


# In[41]:


#Let's plot the growth comparision

sns.lineplot(x=growth_df.index, y=growth_df['% change'])
plt.title("Monthwise Percent growth trend over last year")
plt.show()


# ### Data Inference 
# 
# - We observed an exponential growth in the month of March 
# - We can also observe almost 100% growth in the JFM'22 quarter.

# # Section 5 - Multiple Regression Analysis

# In[42]:


df.head()


# ### Step 1 : Creating Dummy Variable 
# 

# In[43]:


## Creating Dummy Variable 

# Step : Let's create new dataframe for ML Analysis

df_ml = df.copy()
df_ml


# In[44]:


# Let's create dummy for the variables 

dummy_var = [i for i in df_ml.select_dtypes(include='object').columns]

for i in dummy_var:
    dummy_list = pd.get_dummies(data = df_ml[i], drop_first = True)
    df_ml = pd.concat([df_ml, dummy_list], axis =1). drop(labels=i, axis=1)


# In[45]:


df_ml.head(5)


# ###### Data Inference : Our new dataframe has 730 rows and 29 columns.

# In[46]:


###Plotting the heatmap for the new data

plt.figure(figsize = (25,15))
plt.title("Heatmap for the new data frame after dummy creation".title(), fontsize =18)
sns.heatmap(df_ml.corr(), annot = True, cmap = "Spectral")
plt.show()


# ### Data Inference 
# 
# - We observed an densly populated heatmap after creating dummy values 
# 

# ### Step 2 : Train-Test Split
# 

# - Here we will be dividing the data into 2 parts 
#    - Train Data - on which the model will be built, 70% of the data set
#    - Test set : To test the trained model
#           
# 

# In[47]:


df_ml.head()


# In[48]:


# split

df_train, df_test = train_test_split(df_ml, train_size=0.7, random_state=100)
print(f' Training set : {df_train.shape}')
print(f'Test set : {df_test.shape}')


# In[49]:


df_train.sample(5)


# In[50]:


df_test.sample(5)


# ### Step 3 :  Rescaling of the data 
# 

# - We have 2 techniques of rescaling techniques : 
#     - Normalization  : It compresses the features in the range between 0 and 1 
#     - Standadization : It moves data on plot in a way that mean becomes 0 and standard deviation becomes 1

# In[51]:


### Rescaling the data is done through MinMax Scaler 

scaler = MinMaxScaler()


# In[52]:


# List for scaling
scaler_list = [i for i in df_train.columns if df_train[i].nunique() > 2]

df_train[scaler_list] = scaler.fit_transform(df_train[scaler_list])


# In[53]:


df_train.head(5)


# ### Let's define Target and Feature Variable for Modeling Process
# 

# In[54]:


#y will only be target varibale

y_train = df_train.pop('cnt')

#X will be all other variables except target variable

X_train = df_train


# In[55]:


print(f'X_train.shape :{X_train.shape}')
print(f'y_train.shape :{y_train.shape}')


# In[56]:


X_train.head(2)


# ## Step 3 Model Development 
# 
# - Automated Selection : By using RFE
# - Manual Selection : After using automated selection, we will use manual approach in order to acheieve the desired model

# ### Recursive Feature Elimination (RFE)

# - Since we have 29 variables to predict, let's use Recursive Feature Elimination (RFE) method which will help us in ranking the variables in the order of significance/importance

# In[57]:


# Creating object for linear Regression
lm = LinearRegression()


# In[58]:


lm.fit(X_train, y_train)

rfe = RFE(lm,15)

rfe = rfe.fit(X_train, y_train)


# In[59]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# ### Data Inference 
# 
# - RFE helped us to get the top 15 feature 
# - Where RFE.support is True, the feature must be included in the model 
# 

# #### Using above results to extract 15 significant variables

# In[60]:


# let's create a temporary dataframe 
temp_rfe_df = pd.DataFrame()

# Adding name of features: 
temp_rfe_df['Col'] = X_train.columns

temp_rfe_df['RFE_Support'] = rfe.support_

temp_rfe_df = temp_rfe_df[temp_rfe_df['RFE_Support']== True]

temp_rfe_df


# In[61]:


# let's create a list for temp_rfe_df

cols = [i for i in temp_rfe_df['Col']]

X_train[cols].head()


# In[62]:


X_train[cols].shape


# In[63]:


print (f'Final df for model training has {X_train[cols].shape}' )


# ### Model Development Stage : 
# 
# - Variance Influence Factor : From statsmodel library, this will help in dealing with multicolinearity.
#    - If the VIF>5, we will drop those variables
# 
# - Ordinary Least Squares : From statsmodel library, this will help in minimisinf the resuidual squares to detemind
#     - If the p-value > 0.05, we will drop those variables 
# 
# 

# ##### Follwing steps will performed now : 
# - We will define functions - VIF and OLS
# - We will start with all 15 variables at the initial, calculating VIF for each variables 
# - After observing combination of VIF values and p-values generated by OLS model, will go ahead and drop isignificant variables
# - After dropping each variable, an increase in R square and decrease in p value is expected to be observed
# - Once all insignificant variables are dropped, we will have our final model

# In[64]:


#### Defining VIF

def fetch_vif_df(local_df):
    vif_df = pd.DataFrame()
    vif_df['Features'] = local_df.columns
    vif_df['VIF'] = [variance_inflation_factor(local_df.values, i) for i in range(local_df.shape[1])]
    vif_df['VIF'] = round(vif_df['VIF'],2)
    vif_df = vif_df.sort_values(by='VIF', ascending =False)
    vif_df = vif_df.reset_index(drop=True)
    return vif_df


# In[65]:


#### Defining Regression Statistics 

def regmodel_ols(y_dataframe, X_dataframe):
    
    X_dataframe = sm.add_constant(X_dataframe)
    
    lm = sm.OLS(y_dataframe, X_dataframe).fit()
    
    print(lm.summary())


# In[66]:


X_train_df = X_train[cols]
X_train_df


# ### Building the Regression model

# ### Model 1 

# In[67]:


regmodel_ols(y_train, X_train_df)


# In[68]:


#Observing VIF values: 

fetch_vif_df(X_train_df)


# ### Data Inference 
# 
# - The p-values for hum is very high and for rest variable it is under threshold 
# - Humidity is highly correlated as it has VIF value. Hence we will go ahead and drop humidity( hum variable)

# ### Model 2

# In[69]:


## Dropping 'hum' variable 

X_train_1 = X_train_df.drop(labels ='hum', axis =1)

regmodel_ols(y_train, X_train_1)


# In[70]:


fetch_vif_df(X_train_1)


# ### Data Inference 
# 
# - The p-values for Summer, Nov, Dec is beyond the threshold. We will drop 'summer'
# - The VIF values for temp is above the threshold 

# ### Model 3 

# In[71]:


## Dropping 'Summer' variable 

X_train_2 = X_train_1.drop(labels ='Summer', axis =1)

regmodel_ols(y_train, X_train_2)


# In[72]:


fetch_vif_df(X_train_2)


# ### Data Inference 
# - After dropping Summer,a minimal impact has been observed on the R-squared and adj R-squared.
# - The p-value for Nov is still high . Hence we will drop Nov
# 
# 

# ### Model 4

# In[73]:


## Dropping 'Nov' variable 

X_train_3 = X_train_2.drop(labels ='Nov', axis =1)

regmodel_ols(y_train, X_train_3)


# In[74]:


fetch_vif_df(X_train_3)


# ### Data Inference 
# - After dropping Nov,a slight decrease has been observed on the R-squared and adj R-squared.
# - The p-value for Dec seems a little near to threshold. Hence we will drop Dec
# 
# 

# ### Model 5 

# In[75]:


## Dropping 'Dec' variable 

X_train_4 = X_train_3.drop(labels ='Dec', axis =1)

regmodel_ols(y_train, X_train_4)


# In[76]:


fetch_vif_df(X_train_4)


# ### Data Inference 
# - After dropping Dec,a nominal decrease has been observed on the R-squared and adj R-squared.
# - The p-value for Jan seems high however the VIF is below the decided threshold. Hence we will drop Jan and observe
# 
# 

# ### Model 6

# In[77]:


## Dropping 'Dec' variable 

X_train_5 = X_train_4.drop(labels ='Jan', axis =1)

regmodel_ols(y_train, X_train_5)


# In[78]:


fetch_vif_df(X_train_5)


# ### Data Inference 
# - After dropping Jan, all the p-values are below the decided threshold.
# - All VIF values are under the decided threshold

# ###### Hence, all VIF values are >5 and contributing to assumption of multicolinearity and making the model correct.

# ### Residual Analysis

# #### Normal Distribution of Errors 
# 

# In[79]:


lm = sm.OLS(y_train, X_train_5).fit()
y_train_pred = lm.predict(X_train_5)
residuals = y_train-y_train_pred


# In[80]:



plt.figure(figsize=(20,5))

sns.regplot(y=residuals.values, x=y_train_pred.values ,
            scatter_kws = {'color' : "purple"} ,line_kws = {'color': "black"})
plt.title('Residual Vs Prediction', fontsize = 20)

plt.show()


# In[81]:


plt.figure(figsize=(10,7))

sns.distplot(residuals, bins=10, color = 'Green')
plt.title('Error Distribution', fontsize = 20)

plt.show()


# ### Data Inference 
# - We observe mean is close to 0
# - For small prediction, the residuals are on higher side and above mean while for larger prediction , it is aligned
# - Hence, validating our error distribution mean around 0

# ### Testing Homoscedasticity

# In[82]:


residuals = y_train-y_train_pred

plt.figure (figsize = (10,5))

sns.regplot(y =y_train, x =y_train_pred ,
            scatter_kws = {'color' : "grey"} ,line_kws = {'color': "black"})
plt.title('Y train Vs Predictions'.title(), fontsize = 20)
plt.ylabel("y_train".title(), fontsize= 12)

plt.show()


# ### Data Inference 
# - The plot above shows an almost constant variance of prediction.
#     Hence, validating our assumption of Homoscedasticity 

# ### Testing for correlation between Error Terms

# In[83]:


residuals = y_train-y_train_pred

plt.figure(figsize = (10,5))

sns.scatterplot(y=y_train, x=residuals)

plt.title('y_train values Vs Predictions'.title(), fontsize=20)
plt.ylabel('y_train')
plt.show()


# ## Making Predictions 

# In[84]:


### Applying Scaling on Test Dataset 

df_test.sample(5)


# In[85]:


df_test[scaler_list] = scaler.transform(df_test[scaler_list])

#Since Scaler is already fitted, hence we will not perform fit() or fit_tranform() step


# In[86]:


df_test.sample(5)


# In[87]:


#Spliting the data into X_test and y_test

y_test = df_test.pop('cnt')
X_test = df_test


# In[88]:


y_test.shape


# In[89]:


X_test.shape


# In[90]:


### Making Actual Predictions
X_test_new = X_test[X_train_5.columns]


# In[91]:


y_test_pred = lm.predict(X_test_new)


# In[92]:


### Evaluating the Predictions


# In[93]:


fig = plt.figure(figsize = (10,5))

sns.regplot(x=y_test, y=y_test_pred ,
           scatter_kws={'color': 'grey'}, line_kws={'color' : 'black'})

plt.title("Actual Vs Predicted Y test", fontsize= 20)
plt.ylabel("predicted y test values", fontsize = 14)
plt.show()


# ### Data Inference 
# - We observe a strong visual represntation, hence our prediction can be considered as a healthy fit
# - We observe very few outliers, but on a broader terms, the trend depicts a healthy fit

# In[94]:


### Model Quality


# In[95]:


print ('R2 sq of test data predictions: ',  round(r2_score(y_pred=y_test_pred, y_true = y_test),5))
print ('R2 sq of train data predictions: ', round(r2_score(y_pred=y_train_pred, y_true = y_train),5))
print ('Difference between test and train :', 
       abs(round(r2_score(y_pred=y_test_pred, y_true = y_test) - 
                 r2_score(y_pred=y_train_pred, y_true = y_train) ,5)))
                             
                             
                             
                             


# ### Data Inference 
# 
# - Difference is less than 5%, making the model fit for generalization

# ### Assignment Question : Which variables are significant in predicting the demand for shared bikes.
# 
# - To answer this, let's derive the equation 
# 

# In[96]:


## Equation of Prediction


# In[97]:


#Const for constant variable, from OLS model of X_train_5

const = 0.2526

parameter_series = pd.Series(lm.params)

print(parameter_series)


# In[98]:


#Equation 

print(f'Demand of bike = {round(const,3)}')
for i in range(len(parameter_series)):
    if i != len(parameter_series)-1:
        print(f'\t{parameter_series.index[i]} x {round(parameter_series.values[i], 3)} + ')
    else:
        print(f'\t{parameter_series.index[i]} x{round(parameter_series.values[i], 3)} ')


# Demand of bike = 0.253 +(Year x 0.242) + (holiday x -0.093) + (temp x 0.731) + (windspeed x -0.023) + (Spring x 0.003) + (Winter x 0.127) + (Jul x -0.101) + (Sept x 0.055) +  (Light Rain or Snow x -0.277 )+ (Misty and Cloudy x-0.059)

# ### Equation Inference
# 
# - Coeficient of year, temp , spring, winter, september are positive
# - Coeficient of holiday, July windspeed, light rain or snow are negative

# ### Equation Inference
# 
# ###### Positive Impact
# 
# - If all variables are kept 0, the demand for bikes = 0.253
# - If all variables remain constant, we can expect the demand for bikes to increase by 0.242 every year
# - If all variables remain constant, we can expect the demand for bikes to increase by 0.003 in spring months
# - If all variables remain constant, we can expect the demand for bikes to increase by 0.127 in winter months
# - If all variables remain constant, we can expect the demand for bikes to increase by 0.055 in Sept month.
# 
# ##### Negative Impact 
# 
# - If all variables remain constant, we can expect the demand for bikes to decrease by 0.101 in July month
# - If all variables remain constant, we can expect the demand for bikes to decrease by 0.277 in weather situation of Light Rain or Snow
# - If all variables remain constant, we can expect the demand for bikes to decrease by 0.059 in weather situation of Mist and cloudy
# 

# ### Assignment Question : How well those variables describe the bike demands
# 
# - To answer this, let's define a function and plot a graph against strength of the variables

# In[99]:


## Defining a function to predict top 'n' featires
def nImportatnFeatures(series,n):
    series = series.sort_values(key=lambda x : abs(x), ascending = False)
    return series.head(n)


# In[100]:


Descending_features = nImportatnFeatures(parameter_series, len(parameter_series))
Descending_features


# ### Plotting the Features against their order of Strength

# In[101]:


plt.figure(figsize = (15,7))

sns.barplot(x=Descending_features.index, y=Descending_features.values, palette="Spectral")
plt.title ( "Features in Descending Order of the Strength", fontsize=20)
plt.show()


# ## Top 5 Predictors
# 
# - We currently have too many variable, which may make the business decisions complex. 
# Let's go ahead and pick the top 5 predictors. 

# In[102]:


top_5_pred = nImportatnFeatures(parameter_series, 5)
top_5_pred


# In[103]:


plt.figure(figsize = (10,7))

sns.barplot(x=top_5_pred.index, y=top_5_pred.values, palette="Spectral")
plt.title ( "Top 5 Predictors", fontsize=20)
plt.show()


# In[104]:


# Let's narrow down to top 3 predictions
top_3_pred = nImportatnFeatures(parameter_series, 3)
top_3_pred


# ### Business Recommendations 
# 
# - New offer can be release in the month of Spring, to boost the sale
# - The business growth plan can be execured in the September.
# - Expect a stagnancy or degrowth during weathersituations like light rain or snow.

# Submitted by : Sonal Parmar (DSC 41 Batch)
