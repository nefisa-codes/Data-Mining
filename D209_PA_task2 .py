#!/usr/bin/env python
# coding: utf-8

# Nefisa Hassen D209 PA Task 2 

# In[ ]:


pip install --upgrade scikit-learn


# # Part III: Data Preparation

# In[5]:


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


med_data = pd.read_csv('medical_clean.csv') #importaing the data


# In[7]:


med_data.columns #looking at the varialbles 


# In[8]:


#deleting the columns that are irrelevant to answering the research question.


# In[9]:


med_data = med_data.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID','Job', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng','Population', 'Area', 'TimeZone' , 'Item1', 'Item2', 'Item3', 'Item4',
       'Item5', 'Item6', 'Item7', 'Item8' ])


# In[10]:


med_data.columns #checking to bee the columns are deleted 


# In[11]:


med_data.dtypes #looking at data types for each variables 


# In[12]:


med_data.isnull().sum() # cheking for missing data


# In[13]:


#creating dummy variables 


# In[14]:


categorical_columns = [ 
    'ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
    'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
    'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma'
]

for column in categorical_columns:
    med_data[column] = med_data[column].astype('category').cat.codes


# In[15]:


#one hot encoding for catagorical variables that have more than 2 options 


# In[16]:


med_data = pd.get_dummies(med_data, columns=['Services','Complication_risk','Initial_admin'], drop_first=True)


# In[17]:


# checking for outliers. 
med_data.std()


# In[ ]:


#treating outliers with zscore method. 


# In[ ]:


med_data ['TotalCharge_z']=stats.zscore(med_data['TotalCharge'])


# In[ ]:


med_data_outliers_TotalCharge = med_data.query('TotalCharge_z > 3 | TotalCharge_z< -3')


# In[ ]:


med_data ['Additional_charges_z']=stats.zscore(med_data['Additional_charges'])


# In[ ]:


med_data_outliers_Additional_charges  = med_data.query('Additional_charges_z > 3 | Additional_charges_z< -3')


# In[ ]:


med_data ['Initial_days_z'] = stats.zscore(med_data['Initial_days'])


# In[ ]:


med_data_outliers_Initial_days = med_data.query('Initial_days_z > 3 | Initial_days_z< -3')


# In[ ]:


med_data ['VitD_levels_z'] = stats.zscore(med_data['VitD_levels'])


# In[ ]:


med_data_outliers_VitD_levels = med_data.query('VitD_levels_z > 3 | VitD_levels_z< -3')


# In[ ]:


med_data.std() # checking to see if outliers were treated.


# In[ ]:


med_data = med_data.astype(int)


# In[ ]:


med_data['VitD_levels'].describe()


# In[ ]:


med_data['Doc_visits'].describe()


# In[ ]:


med_data['Full_meals_eaten'].describe()


# In[ ]:


med_data['vitD_supp'].value_counts()


# In[ ]:


med_data['Initial_admin_Emergency Admission'].value_counts()


# In[ ]:


med_data['Initial_admin_Observation Admission'].value_counts()


# In[ ]:


med_data['Complication_risk_Low'].value_counts()


# In[ ]:


med_data['Complication_risk_Medium'].value_counts()


# In[ ]:


med_data['Services_CT Scan'].value_counts()


# In[ ]:


med_data['Services_Intravenous'].value_counts()


# In[ ]:


med_data['Services_MRI'].value_counts()


# In[ ]:


med_data['Overweight'].value_counts()


# In[ ]:


med_data['Arthritis'].value_counts()


# In[ ]:


med_data['Diabetes'].value_counts()


# In[ ]:


med_data['Hyperlipidemia'].value_counts()


# In[ ]:


med_data['BackPain'].value_counts()


# In[ ]:


med_data['Anxiety'].value_counts()


# In[ ]:


med_data['Allergic_rhinitis'].value_counts()


# In[ ]:


med_data['Reflux_esophagitis'].value_counts()


# In[ ]:


med_data['Asthma'].value_counts()


# In[ ]:


med_data['TotalCharge'].describe()


# In[ ]:


med_data['Additional_charges'].describe()


# In[ ]:


med_data['Asthma'].value_counts()


# In[ ]:


#Describtibve analysis 
med_data.describe()


# In[ ]:


med_data.hist(bins=20, figsize=(15, 10)) 
plt.suptitle('Histograms of all Variables', x=0.5, y=0.92, fontsize=16)
plt.show()


# In[ ]:


correlation_matrix = med_data.corr()


# In[ ]:


plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap of Correlation Matrix fot all the Variables', fontsize=16)
plt.show()


# In[ ]:


#create a dataframe with all training data except the target column
X = med_data.drop(columns = ['ReAdmis'])


# In[ ]:


y = med_data['ReAdmis']


# In[ ]:


skbest = SelectKBest(score_func=f_classif, k='all') 

X_new=skbest.fit_transform(X,y) 

X_new.shape 


# In[ ]:


p_values = pd.DataFrame({'Feature': X.columns, 'p_value': skbest.pvalues_}).sort_values('p_value') 
p_values_filtered = p_values[p_values['p_value'] < 0.05] 
features_to_keep = p_values['Feature'] 
[p_values['p_value']<.05] 
print(features_to_keep) 


# In[ ]:


vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)


# In[ ]:


Med_data_updated = med_data.drop(['Doc_visits', 'Initial_days', 'Initial_admin_Emergency Admission'], axis=1)


# In[ ]:


def vif_scores(Med_data_updated):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independant Features"] = Med_data_updated.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(Med_data_updated.values,i) for i in range(Med_data_updated.shape[1])]
    return VIF_Scores


# In[ ]:


print(vif_scores(Med_data_updated)) # multicoliarity treated


# In[ ]:


scale = StandardScaler()


# In[ ]:


x_scaled = med_data[['ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
       'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis',
       'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety',
       'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma', 'Initial_days',
       'TotalCharge', 'Additional_charges', 'Services_CT Scan',
       'Services_Intravenous', 'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission', 'TotalCharge_z',
       'Additional_charges_z', 'Initial_days_z', 'VitD_levels_z']]


# In[ ]:


scaled= scale.fit_transform(x_scaled)


# In[ ]:


print(x_scaled)


# In[ ]:


#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# In[ ]:


#C4.  Provide a copy of the cleaned data set.


# In[ ]:


X_train.to_csv('X_train.csv',index=False)
X_test.to_csv('X_test.csv', index=False)


# In[ ]:


y_train_df = pd.DataFrame({'ReAdmis': y_train})
y_train_df.to_csv('y_train.csv', index=False)


# In[ ]:


y_test_df = pd.DataFrame({'ReAdmis': y_test})
y_test_df.to_csv('y_test.csv', index=False)


# # Part IV: Analysis

# In[ ]:


#D1.  Split the data into training and test data sets and provide the file(s).
#the above code shows the data split  into test and train  sets


# In[ ]:


#D2. Describe the analysis technique you used to appropriately analyze the data


# In[ ]:





# In[ ]:


from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params_dist = {
 'max_depth': [4, 6, 8, 10, 12, 14, 16],
 'max_leaf_nodes': [1000, 2000, 3000],
 'min_samples_leaf': [20, 30, 40, 50],
 'min_samples_split': [30, 40, 50]
}


# In[ ]:


dtr = DecisionTreeClassifier()


# In[ ]:


random_search = RandomizedSearchCV(dtr, params_dist, cv=5)


# In[ ]:


random_search.fit(X_train, y_train)


# In[ ]:


print("Best Parameters:", random_search.best_params_)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


y_pred = random_search.predict(X_test)


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# TP: 1243 
# FP: 23 
# FN: 22 
# TN: 712 

# In[ ]:


class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# the model shows high precision, recall, and F1-score for both classes, indicating good performance. 
# The overall accuracy is also high at 98%, which suggests that the model is doing well in classifying the test data

# In[ ]:


#D3.


# In[ ]:


med_data.to_csv('MSDA209_PA_task2_D3.cvs')


# # Part V: Data Summary and Implications

# In[ ]:


#E1.Explain the accuracy and the mean squared error (MSE) of your prediction model.


# In[ ]:


print("Training Score (MSE):", random_search.best_score_)

print ("Training Score (RMSE):",(random_search.best_score_)**(1/2))

y_train_pred = random_search.predict(X_train)
print ("Train - R squared score for the model:",r2_score(y_train,y_train_pred))


# In[ ]:


#  pridiction on the testing data


# In[ ]:


mse_test = mean_squared_error(y_test, y_pred)
print("Testing - Mean squared error for the model:",mse_test )

rmse_test = mse_test ** 0.5

print("Testing -Root Mean Squared error for the model:",rmse_test)

print("Testing - R - squared score for the model:", r2_score(y_test,y_pred))


# In[ ]:


#E2. Discuss the results and implications of your prediction analysis. 


# Training  Data Results: The high R-squared score on the training data (0.9123) suggests that the Decision Tree Classifier captures a significant portion of the variance in the hospital readmission rates within the training dataset. The low MSE and RMSE values further indicate a good fit to the training data. 
# 
# Testing Data Results: The testing results mirror the training results, indicating that the Decision Tree Classifier generalizes well to unseen data. The high R-squared score on the testing data (0.9031) suggests that the model effectively explains the variability in hospital readmission rates. 
# 
# Based on the provided code, the Decision Tree Classifier exhibits strong predictive capabilities for hospital readmission rates. The model performs well on both the training and testing datasets, suggesting its potential utility in predicting readmission rates for new patient cases. 
# 
#  

# In[ ]:


#E3.  Discuss one limitation of your data analysis. 


# The effectiveness of the model heavily relies on the quality and representativeness of the dataset. If the data is incomplete, biased, or contains outliers, the model's performance may be affected. 

# In[ ]:


#E4.  Recommend a course of action for the real-world organizational situation from part A1.


# Based on the analysis I recommend that the organization can leverage the promising results of the Decision Tree Classifier to make informed decisions regarding hospital readmission rates. I highly recommend Regular monitoring of the data, collaboration with healthcare professionals to incorporate feedback to refine and improve the model will contribute to the successful implementation and utilization of the predictive model in a real-world healthcare setting.

# In[ ]:




