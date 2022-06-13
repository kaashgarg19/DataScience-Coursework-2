# Importing Libraries
import numpy as np
import pandas as pd
 
# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')

#for  Visualisation the data
import matplotlib.pyplot as plt
import seaborn as sns


#for preprocessing  the data
from sklearn.preprocessing import  StandardScaler, LabelEncoder

#  for Modelling Helping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , cross_val_score

# Applying ML Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error




#We will predict the selling price of a car based on its various features available in dataset such as kilometeres used, year of buying, Car name, fuel type etc.
#Let us first read the datset and find out various features.
# Reading dataset
df= pd.read_csv("data.csv")
print(df)

# Read the data how dataset looks
df.head()

# according to above we find:
#Vaious features and their meanings:
    #Car_Name: This column belongs to the name of the car.
    #Year: This column belongs to the  year in which the car was bought.
    #Selling_Price: This column should be filled with the price the owner wants to sell the car.
    #Present_Price: This is the current ex-showroom price of the car
    #Kms_Driven: This is the distance completed by the car in km.
    #Fuel_Type: Fuel type of the car.
    #Seller_Type: Defines whether the seller is a dealer or an individual.
    #Transmission: Defines whether the car is manual or automatic.
    #Owner: Defines the number of owners the car has previously had.
    
#  calculating Rows and columns in dataset
df.shape
rw,clm= df.shape
print(f'There are {rw} rows and {clm} columns in our cars dataset.')    
    
#There are 301 rows and 9 columns in our cars dataset.


# Checking any null entry in dataset
nullvalue = df.isnull().any()
nullvalue

#Find Missing Values
missing_values=df.isnull()
print(missing_values)

#It means there is no null value in the cars dataset.

#To print Top 5 rows
print(df.head(5))

#To print bottom  5 rows
print(df.tail(5 ))

#finding additional information aboout data
df.info()
    

# checking datatypes
df.dtypes


#Describe/ summarize the data
#The features described in the above data set are:
#1. Count tells us the number of NoN-empty rows in a feature.
#2. Mean tells us the mean value of that feature.
#3. Std tells us the Standard Deviation Value of that feature.
#4. Min tells us the minimum value of that feature.
#5. 25%, 50%, and 75% are the percentile/quartile of each features.
#6. Max tells us the maximum value of that feature.
describe=df.describe().T
print(describe)

df.describe(include=object)

#Calculating value types of different columns

df['Fuel_Type'].value_counts()
#Output:
#Petrol    239
#Diesel     60
#CNG         2
#Name: Fuel_Type, dtype: int64

df['Seller_Type'].value_counts()

#Dealer        195
#Individual    106
#Name: Seller_Type, dtype: int64

df['Transmission'].value_counts()

#Manual       261
#Automatic     40
#Name: Transmission, dtype: int64

df['Seller_Type'].dtypes

#dtype('O')


# Now, Converting Fuel_Type,Seller Type and Transmission Columns from Object to numeric datatypes to make it machne readable form.

label = LabelEncoder()
df["Fuel_Type"]=label.fit_transform(df["Fuel_Type"])
df['Seller_Type']=label.fit_transform(df['Seller_Type'])
df["Transmission"]=label.fit_transform(df["Transmission"])
 


#Out: 
#car_Name          object
#Year               int64
#Selling_Price    float64
#Present_Price    float64
#Kms_Driven         int64
#Fuel_Type          int64
#Seller_Type        int64
#Transmission       int64
#Owner              int64
#dtype: object


#Data Analysis and Visualization

#Let us analyse various columns and their relationship with selling prices of cars.

df.columns

# Correlation between various columns

ax=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(ax,cmap="Blues",vmin=0, vmax=1)



#This means there is a strong corelation between Selling Price and Present Price of Cars.
# Further, a less stronger relationship between Present Price- Kilometers driven, selling price-year of buying that car and so on.
# There is very less correlation of Selling price with Fuel_type,Seller_Type and Owner.

# Let us ckeck relationship of some selected columns with the Selling Price

car=df[['Year','Selling_Price','Present_Price','Kms_Driven']]
sns.pairplot(car)

#As we can see from graph, Year, Kms_Driven has mahor impact on Selling Price. 
#The latest model cars are having higher Selling prices.
#Similarly, the cars with higher kilometers run tend to have low Selling prices.



#Plotting relationship between Selling Price vs. Present Price
plt.figure(figsize=(8,5))
x=df['Selling_Price']
y=df['Present_Price']

plt.title('Selling Price vs. Present Price')
sns.scatterplot(x,y,color='c')

#It means the selling price of cars is always less than their respective present showroom prices


#Plotting relationship between Selling Price vs. Year

plt.figure(figsize=(8,5))
x=df['Selling_Price']
y=df['Year']

plt.title('Selling Price vs Year')
sns.scatterplot(x,y,color='r')

#The cars bought after year-2010 have higher selling prices than older models of cars.


#Plotting relationship between Selling Price vs. Kilometers Driven

plt.figure(figsize=(8,5))
y=df['Selling_Price']
x=df['Kms_Driven']

plt.title('Selling Price vs. Kilometers Driven')
sns.scatterplot(x,y,color='y')

#This is clearly indicating that the cars that are driven more are selling at lower prices.

#Plotting relationship between Selling Price vs. Kilometers Driven

plt.figure(figsize=(8,5))
x=df['Selling_Price']
y=df['Fuel_Type']

plt.title('Selling Price vs. Fuel type')
sns.scatterplot(x,y,color='c')

#It means we have maximum Selling price and current price of cars are of fuel type-2(i.e Petrol Cars).


# Selling Price vs Fuel Type and Transmission Type

plt.figure(figsize=(10,10))
x= df['Selling_Price']
y= df['Fuel_Type']
x1= df['Selling_Price']
y1= df['Transmission']
sns.jointplot(x,y,data=df,kind='hex',color='yellow')
sns.jointplot(x1,y1,data=df,kind='hex')


#Most of the cars belong to type-2 fuel category(ie. Petrol Cars) and selling prices vary from 1- 7 lacs.
# This means there is not a strong relation between type of cars and selling prices. 
 #If we compare transmission w.r.t Selling price, there is a positive relation.
 #Manual cars(Transmission=0) are having more selling prices than others.
 
 
 # Selling Price vs Seller_Type

plt.figure(figsize=(10,10))
x= df['Selling_Price']
y= df['Seller_Type']
sns.jointplot(x,y,data=df,kind='reg',color='yellow')

#It means the number of cars sold from Seller Type-0(i.e Dealers) are having more Selling prices than others from Individual car owners.

#Plotting relationship between Selling Price vs Number of Owners

plt.figure(figsize=(8,5))
x=df['Selling_Price']
y=df['Owner']

plt.title('Selling Price vs Owner')
sns.scatterplot(x,y,color='g')

#This means the cars having less owners or we can say 1 owner have higher resale values or Selling Price than others.

# calculating most sell and top sell car by car name.
most_sell_cars= df.groupby(['Car_Name']) ['Selling_Price'].nunique().sort_values(ascending=False).head(10)
most_sell_cars

top_sellpric_cars= df.groupby(['Car_Name']) ['Selling_Price'].sum().sort_values(ascending=False).head(10)
top_sellpric_cars

# Plotting relationship between Various cars w.r.t their selling counts and selling price. 
sns.set_style("darkgrid")
sns.set_context("notebook")
plt.figure(figsize=(15,12))

x=most_sell_cars.index
y=most_sell_cars.values

plt.subplot(2,1,1)
plt.bar(x,y,color='r')
plt.xlabel('Car Name')
plt.ylabel('No. of Cars sold')
plt.title('Mostly Sold Cars')

x1=top_sellpric_cars.index
y1=top_sellpric_cars.values

plt.subplot(2,1,2)
plt.bar(x1,y1,color='b')
plt.xlabel('Car Name')
plt.ylabel('Selling Price earned(Lacs)')
plt.title('Top Selling Price Cars')



#If we compare both the plots, the number of cars sold and the highest selling prices cars are almost the same.
 #Look at the top 5 records: Fortuner,City,Innova,Corolla Altis and Verna.
 
 
 
# Now we check whether there is any reduction of Selling price w.r.t Present Showrrom Prices.
df.columns

df['Price Reduction']= df['Present_Price']-df['Selling_Price']
 # Inserting a new column of reduced price
top_reduc= df.groupby('Car_Name')['Price Reduction'].mean().sort_values(ascending=False).head(20)
top_reduc

#This means there reduction in some cars prices.

#Now, we can analyse for these top cars having higher price depreciation.

df['Car_Name'].value_counts()


#Now, we check is there any Price Depreciation in different columns Other Columns


plt.figure(figsize=(15,10))
y1=df['Price Reduction']
x1=df['Owner']
plt.subplot(221)
plt.title('Price Reduction vs No. of Car Owners')
sns.barplot(x1,y1,color='y')

y2=df['Price Reduction']
x2=df['Fuel_Type']
plt.subplot(222)
sns.barplot(x2,y2,color='c')
plt.title('Price Reduction vs Fuel Type')

y3=df['Price Reduction']
x3=df['Seller_Type']
plt.subplot(223)
plt.title('Price Reduction vs Seller Type')
sns.barplot(x3,y3,color='r')


y4=df['Price Reduction']
x4=df['Transmission']
plt.subplot(224)
sns.barplot(x4,y4)
plt.title('Price Reduction vs Transmission')

#Here, we can find following obervations:

#The cars with more number of owners are having large reduction of Selling Price.
#Cars with Fule Type-1(i.e Diesel Cars) have higher drepreciaiton in Selling Price.
#Seller Type-0(i.e. Cars sold from Dealers have higher depreciation in Selling Price.
#The cars with automatic transmission(i.e. value=1) have higher drepreciation in Selling Price.

#We drop the Car_Name.
#We will take the numerical features as it is.


pair_df = [df[["Year", "Present_Price", "Kms_Driven", "Owner"]], 
           pd.get_dummies(df[["Fuel_Type", "Seller_Type", "Transmission"]], drop_first=True), df[["Selling_Price"]]]
X = pd.concat(pair_df, axis=1)
y = df[["Selling_Price"]]

# Lets have a look into processed data
X.head()
	
# Dependent variable
y.head()

#Applying Algorithms
#Let us apply varrious algorithm to find out the selling price of a car based various column features.
#As our datset is a regression dataset, so we can apply various regression algoirthm to solve our problem 
#suc as: Linear Regression,Decision Tree and Random Forest.




# Let's delete the Selling_Price from X
X.drop(labels=["Selling_Price"], axis=1, inplace=True)

# Applying Feature Scaling 
#( NormalizationStandardScaler )

sc = StandardScaler()
X = sc.fit_transform(X)

#Spliting data to train and test sizes.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
print(X_test)
print(y_test)

# Shape of the dataset

X_train.shape,X_test.shape,y_train.shape,y_test.shape

#((210, 7), (91, 7), (210, 1), (91, 1))

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



Scores=[]

# Applying Linear Regression

linreg = LinearRegression()
linreg.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# MSE on training data

mean_squared_error(y_true=y_train, y_pred=linreg.predict(X_train))

#Output:3.4416817237822563

# MAE on training data 


mean_absolute_error(y_true=y_train, y_pred=linreg.predict(X_train))

#Output: 1.2588516835383443

# R squared on training data

# This method returns R squared value
# This will first predict the values for X_train since our model is already fit and ten will calculate the R^2 value

linreg.score(X_train, y_train)

#Output:0.8863492418513184
# We will get the same value with r2_score function also

r2_score(y_true=y_train, y_pred=linreg.predict(X_train))

#R2_Score: 0.8863492418513184

#Linear regression with cross validation cross
lr = LinearRegression()
scores= cross_val_score( lr, X = X_train, y = y_train, cv = 10)
print("Cross-Validation scores:{}".format(scores))
print("Average Cross-Validation score:{:.2f}".format(scores.mean()))

#Cross-Validation scores:[0.79141458 0.9317119  0.47072961 0.7863168  0.87100179 0.91764002
# 0.7231416  0.78236958 0.75426197 0.92438874]
#Average Cross-Validation score:0.80


# Lets check the metrics on test data
mse = mean_squared_error(y_true=y_test, y_pred=linreg.predict(X_test))
mae = mean_absolute_error(y_true=y_test, y_pred=linreg.predict(X_test))
rmse = mean_squared_error(y_true=y_test, y_pred=linreg.predict(X_test))**0.5
r2 = linreg.score(X_test, y_test)

print("MSE on test data: ", mse)
print("MAE on test data: ", mae)
print('RMSE   : %0.2f ' % rmse)
print("R squared on test data: ", r2)

Scores.append(scores)

#MSE on test data:  2.608704211216183
#MAE on test data:  1.194677003868084
#R squared on test data:  0.8191909743726961

#plotting result of Actual and Predicted values
plt.scatter(y_test, linreg.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# Decision Tree Regression with cross validation

Dt =DecisionTreeRegressor()
Dt.fit(X_train , y_train)
scores1 = cross_val_score(estimator = Dt, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = Dt.predict(X_test)

print('Decision Tree Regression')

      
print('Average Cross-Validation score : %.4f' % Dt.score(X_test, y_test))
print(scores1)

#Decision Tree Regression
#Average Cross-Validation score : 0.9405
# cross validation scores:[0.75517547 0.94618767 0.79649486 0.91568565 0.95160693 0.98180857
# 0.95986495 0.95992797 0.96839264 0.95619655]

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print("MSE on test data: ", mse)
print("MAE on test data: ", mae)
print('RMSE   : %0.2f ' % rmse)
print("R squared on test data: ", r2)


Scores.append(scores1)

#MSE on test data:  0.8582593406593407
#MAE on test data:  0.5523076923076923
#RMSE   : 0.93 
#R squared on test data:  0.9405141317084001


#plotting result of Actual and Predicted values
plt.scatter(y_test, Dt.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()



# Random Forest Regression with cross validation

random = RandomForestRegressor()
random.fit(X_train , y_train)
scores2 = cross_val_score(estimator = random, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = random.predict(X_test)
print('.........................')
print(' Random Forest ')
print('Average cross validationScore : %.4f' % random.score(X_test, y_test))
print(scores2)

#Average cross validationScore : 0.9560
#Cross validation score:[0.81393753 0.94472373 0.91257379 0.9424919  0.90455688 0.98350306
# 0.9552075  0.90817853 0.93949919 0.97974876]



mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

#Output
#MSE    : 0.89 
#MAE    : 0.51 
#RMSE   : 0.94 
#R2     : 0.94


#Model Tuning :
#A model hyperparameter is external configuration of model. 
#They are often tuned for a predictive problem. 
#Grid-search is used to find the optimal hyperparameters for more accurate predictions and estimate model performance on unseen data.
# We tried to enhance our performance score by using grid search cv and passing parameters on Random Forest regressor.

no_of_test=[100]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2'],'max_depth':[3,5,7,10],}
random=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2', cv=10)
random.fit(X_train,y_train)
print('Score : %.4f' % random.score(X_test, y_test))
pred=random.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)


Scores.append(scores2)

# output after applying Tuning:
#Score : 0.9573
#R2     : 0.96 

#Best score after tuning parameters:
random.best_score_
# 0.9312184412696669

#plotting result of Actual and Predicted values
plt.scatter(y_test, random.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()


# Finally, Comparison between all models:

Result = pd.DataFrame({'Algorithms': ['Linear Regression' , 'RandomForest Regression' , 'DecisionTree Regression'],
                        'Score': [ linreg.score(X_test, y_test) ,random.score(X_test, y_test), Dt.score(X_test, y_test)]})
result_df = Result.sort_values(by='Score', ascending=False)
result_df = Result.set_index('Score')
print(result_df)


#                   Algorithms
#Score                            
#0.819191        Linear Regression
#0.957327  RandomForest Regression
#0.939210  DecisionTree Regression

#As a Result, It is clear that Random Forest Regressor gives us the Best accuracies score of [95% ]
#after applying Hyperparameter tuning.


ML_models=['Linear Regression' , 'RandomForest Regression' , 'DecisionTree Regression']




#Using Barplot to compare results of R2 Scores
sns.barplot(x='Score' , y='Algorithms' , data= Result)

#Using Boxplot to compare  all models results

fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(Scores)
ax.set_xticklabels(ML_models)
plt.show()

# This boxplot result showing the spread of accuracy scores across each cross-validation fold.



#Using factorplot to compare results of Models:-



sns.factorplot(x='Algorithms', y='Score', data=Result, size=5 , aspect=4)

##########>>>>>>>>>>>>>>>>>>>>>>>>>>>>####################################>>>>>>>>>

# As a rsult,  after comparision of different models we can say Random Forest Regression model is the best model to predict the car selling price by accuracy of 95%. 
