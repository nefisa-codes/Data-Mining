#!/usr/bin/env python
# coding: utf-8

# Nefisa Hassen D208 PA Task 2 

# In[90]:


pip install --upgrade scikit-learn


# # Part III: Data Preparation

# In[96]:


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


# In[97]:


med_data = pd.read_csv('medical_clean.csv') #importaing the data


# In[98]:


med_data.columns #looking at the varialbles 


# In[99]:


#deleting the columns that are irrelevant to answering the research question.


# In[100]:


med_data = med_data.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID','Job', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng','Population', 'Area', 'TimeZone' , 'Item1', 'Item2', 'Item3', 'Item4',
       'Item5', 'Item6', 'Item7', 'Item8' ])


# In[101]:


med_data.columns #checking to bee the columns are deleted 


# In[102]:


med_data.dtypes #looking at data types for each variables 


# In[103]:


med_data.isnull().sum() # cheking for missing data


# In[9]:


#creating dummy variables 


# In[104]:


categorical_columns = [ 
    'ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
    'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
    'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma'
]

for column in categorical_columns:
    med_data[column] = med_data[column].astype('category').cat.codes


# In[105]:


#one hot encoding for catagorical variables that have more than 2 options 


# In[106]:


med_data = pd.get_dummies(med_data, columns=['Services','Complication_risk','Initial_admin'], drop_first=True)


# In[107]:


# checking for outliers. 
med_data.std()


# In[14]:


#treating outliers with zscore method. 


# In[108]:


med_data ['TotalCharge_z']=stats.zscore(med_data['TotalCharge'])


# In[109]:


med_data_outliers_TotalCharge = med_data.query('TotalCharge_z > 3 | TotalCharge_z< -3')


# In[110]:


med_data ['Additional_charges_z']=stats.zscore(med_data['Additional_charges'])


# In[111]:


med_data_outliers_Additional_charges  = med_data.query('Additional_charges_z > 3 | Additional_charges_z< -3')


# In[112]:


med_data ['Initial_days_z'] = stats.zscore(med_data['Initial_days'])


# In[113]:


med_data_outliers_Initial_days = med_data.query('Initial_days_z > 3 | Initial_days_z< -3')


# In[114]:


med_data ['VitD_levels_z'] = stats.zscore(med_data['VitD_levels'])


# In[115]:


med_data_outliers_VitD_levels = med_data.query('VitD_levels_z > 3 | VitD_levels_z< -3')


# In[116]:


med_data.std() # checking to see if outliers were treated.


# In[118]:


med_data = med_data.astype(int)


# In[119]:


med_data['VitD_levels'].describe()


# In[120]:


med_data['Doc_visits'].describe()


# In[121]:


med_data['Full_meals_eaten'].describe()


# In[122]:


med_data['vitD_supp'].value_counts()


# In[123]:


med_data['Initial_admin_Emergency Admission'].value_counts()


# In[124]:


med_data['Initial_admin_Observation Admission'].value_counts()


# In[125]:


med_data['Complication_risk_Low'].value_counts()


# In[126]:


med_data['Complication_risk_Medium'].value_counts()


# In[127]:


med_data['Services_CT Scan'].value_counts()


# In[128]:


med_data['Services_Intravenous'].value_counts()


# In[129]:


med_data['Services_MRI'].value_counts()


# In[130]:


med_data['Overweight'].value_counts()


# In[131]:


med_data['Arthritis'].value_counts()


# In[132]:


med_data['Diabetes'].value_counts()


# In[133]:


med_data['Hyperlipidemia'].value_counts()


# In[134]:


med_data['BackPain'].value_counts()


# In[135]:


med_data['Anxiety'].value_counts()


# In[136]:


med_data['Allergic_rhinitis'].value_counts()


# In[137]:


med_data['Reflux_esophagitis'].value_counts()


# In[138]:


med_data['Asthma'].value_counts()


# In[139]:


med_data['TotalCharge'].describe()


# In[140]:


med_data['Additional_charges'].describe()


# In[141]:


med_data['Asthma'].value_counts()


# In[142]:


#Describtibve analysis 
med_data.describe()


# In[143]:


med_data.hist(bins=20, figsize=(15, 10)) 
plt.suptitle('Histograms of all Variables', x=0.5, y=0.92, fontsize=16)
plt.show()


# In[144]:


correlation_matrix = med_data.corr()


# In[145]:


plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap of Correlation Matrix fot all the Variables', fontsize=16)
plt.show()


# In[146]:


#create a dataframe with all training data except the target column
X = med_data.drop(columns = ['ReAdmis'])


# In[147]:


y = med_data['ReAdmis']


# In[148]:


skbest = SelectKBest(score_func=f_classif, k='all') 

X_new=skbest.fit_transform(X,y) 

X_new.shape 


# In[149]:


p_values = pd.DataFrame({'Feature': X.columns, 'p_value': skbest.pvalues_}).sort_values('p_value') 
p_values_filtered = p_values[p_values['p_value'] < 0.05] 
features_to_keep = p_values['Feature'] 
[p_values['p_value']<.05] 
print(features_to_keep) 


# In[151]:


vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)


# In[152]:


Med_data_updated = med_data.drop(['Doc_visits', 'Initial_days', 'Initial_admin_Emergency Admission'], axis=1)


# In[153]:


def vif_scores(Med_data_updated):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independant Features"] = Med_data_updated.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(Med_data_updated.values,i) for i in range(Med_data_updated.shape[1])]
    return VIF_Scores


# In[59]:


print(vif_scores(Med_data_updated)) # multicoliarity trated


# In[154]:


scale = StandardScaler()


# In[155]:


x_scaled = med_data[['ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
       'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis',
       'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety',
       'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma', 'Initial_days',
       'TotalCharge', 'Additional_charges', 'Services_CT Scan',
       'Services_Intravenous', 'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission', 'TotalCharge_z',
       'Additional_charges_z', 'Initial_days_z', 'VitD_levels_z']]


# In[156]:


scaled= scale.fit_transform(x_scaled)


# In[157]:


print(x_scaled)


# In[158]:


#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# In[159]:


#C4.  Provide a copy of the cleaned data set.


# In[160]:


X_train.to_csv('X_train.csv',index=False)
X_test.to_csv('X_test.csv', index=False)


# In[161]:


y_train_df = pd.DataFrame({'ReAdmis': y_train})
y_train_df.to_csv('y_train.csv', index=False)


# In[162]:


y_test_df = pd.DataFrame({'ReAdmis': y_test})
y_test_df.to_csv('y_test.csv', index=False)


# # Part IV: Analysis

# In[69]:


#D1.  Split the data into training and test data sets and provide the file(s).


# In[70]:


#the above code shows the data split  into test and train  sets


# In[71]:


#D2. Describe the analysis technique you used to appropriately analyze the data


# First, define a parameter grid for n_neighbors ranging from 1 to 51 and create an instance of the KNeighborsClassifier. Secondly, create a GridSearchCV object using the KNN classifier. Fit the GridSearchCV model to the training data, which involves training the KNN classifier with different values of n_neighbors and using cross-validation to find the best hyperparameter. By using the best_params function, it is determined that the optimal number of neighbors to consider for classification is 5. The best_score indicates that the highest cross-validated accuracy is approximately 94.83%.

# In[72]:


param_grid = {'n_neighbors': np.arange(1, 51)}


# In[73]:


knn = KNeighborsClassifier()


# In[74]:


knn_cv = GridSearchCV(knn, param_grid, cv=5)


# In[75]:


knn_cv.fit(X_train, y_train)


# In[76]:


knn_cv.best_params_


# In[163]:


knn_cv.best_score_


# In[164]:


knn = KNeighborsClassifier(n_neighbors =5)


# In[165]:


knn.fit(X_train, y_train)


# In[ ]:





# In[166]:


y_pred = knn.predict(X_test)
y_pred


# In[167]:


final_matrix = confusion_matrix(y_test, y_pred)
final_matrix


# In[82]:


sns.heatmap(final_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#  (TP) 696 - The model correctly predicted that 696 instances belong to class 1 (positive).
#  (TN) 1184 - The model correctly predicted that 1184 instances belong to class 0 (negative).
#  (FP)82 - The model incorrectly predicted that 82 instances belong to class 1.
#  (FN) 38 - The model incorrectly predicted that 38 instances belong to class 0.

# In[168]:


#D3.  Provide the code used to perform the classification analysis from part D2.


# In[169]:


med_data.to_csv('MSDA209_PA_task1_D2.cvs')


# # Part V: Data Summary and Implications

# In[ ]:


#E1.Explain the accuracy and the area under the curve (AUC) of your classification model.


# Based on the provided code, the model demonstrates an accuracy of 94% on the test set. Additionally, the Area Under the Curve (AUC) is calculated to be 0.9417, a value near 1. This suggests strong overall performance, affirming the model's ability to effectively distinguish between positive and negative instances.

# In[170]:


# Calculating  accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')


# In[171]:


# Classification Report
classification_rep = classification_report(y_test, y_pred)
print('\nClassification Report:')
print(classification_rep)


# In[172]:


# Calculating  AUC
accuracy_new = metrics.roc_auc_score(y_test, y_pred)
print(f'AUC: {accuracy_new:.4f}')


# In[173]:


fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc_new = auc(fpr, tpr)


# In[89]:


# Ploting  ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_new)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#E2.  Discuss the results and implications of your classification analysis.


# The model demonstrates high accuracy (94.825%) in a cross-validated setting for predicting patient readmission based on chronic conditions, lifestyle factors, and healthcare services a patient receives. The examination of the classification report further explains the model's notable achievements in terms of high precision, recall, and F1-score. Additionally, the analysis includes the computation of the Area Under the Curve (AUC), a metric ranging from 0 to 1. The calculated AUC of 0.9417, nearing the upper limit, signifies superior model performance. This proximity to 1 implies that the model demonstrates a robust true positive rate and maintains a low false positive rate across diverse threshold settings, highlighting its effectiveness in distinguishing between positive and negative instances.

# In[ ]:


#E3.Discuss one limitation of your data analysis.


# One possible limitation for the model is data quality and bias. The model's performance heavily relies on the quality
# and representativeness of the training data. If the data used for training is biased or incomplete, the model may
# inherit these biases and make inaccurate predictions, especially in underrepresented groups. Another limitation is 
# that the analysis relies on a dataset of 10,000 patients, which might be inadequate for making precise outcome 
# predictions.

# In[ ]:


#E4.Recommend a course of action for the real-world organizational situation from part A1 based on your results and implications discussed in part E2.



#  first,I recommend that the health organization implement the KNN classification model into their system to predict patient readmission based on chronic conditions, lifestyle factors, and healthcare services. This model, with a best-performing parameter of 5 neighbors, has demonstrated high accuracy (94.825%) in a cross-validated setting.
# 
# second.Establish a system for continuous monitoring and evaluation of the model's performance. Regularly assess its accuracy, precision, recall, and other relevant metrics to ensure its effectiveness over time.
# 
#  lastly, Collaborate with healthcare professionals to integrate the predictive model into the existing healthcare workflow. This could involve incorporating the model's predictions into decision-making processes related to patient care and resource allocation.

# In[ ]:


#F


# #G
# Statology. (n.d.). How to Plot an ROC Curve in Python. Retrieved from https://www.statology.org/plot-roc-curve-python/
# 
# Statology. (n.d.). How to Calculate AUC in Python. Retrieved from https://www.statology.org/auc-in-python/
# 
# W3Schools. (n.d.). Python - Machine Learning - Standardize Data. Retrieved from https://www.w3schools.com/python/python_ml_scale.asp
# 
# Pulagam, S. (2020, June 6). How to detect and deal with Multicollinearity. Medium. https://towardsdatascience.com/how-to-detect-and-deal-with-multicollinearity-9e02b18695f1
# 
# Raj, R. (2022, March 16). Classification Algorithms in Python. Thrive in AI. https://medium.com/thrive-in-ai/classification-algorithms-in-python-5f58a7a27b88
# 

# In[ ]:


#I none used 


# In[ ]:


#J


# In[ ]:




