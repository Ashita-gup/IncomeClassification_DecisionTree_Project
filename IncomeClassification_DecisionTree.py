#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Name : Ashita Gupta
# Roll No. : 2022BCY0008
# Lab 9 : Solution 1
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree


# In[33]:


# Loading the Adult Census Income dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv('adult.data', header=None, names=column_names, na_values=' ?', skipinitialspace=True)


# In[34]:


# Data Preprocessing
# Dropping rows with missing values
df.dropna(inplace=True)


# In[35]:


# Encoding categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns


# In[36]:


for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Splitting the data into features (X) and target (y)
X = df.drop(columns=['income'])  # Features
y = df['income']  # Target variable


# In[37]:


# Splitting the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[38]:


# Initializing the Decision Tree Classifier with class weights
best_tree = DecisionTreeClassifier(class_weight={0: 1, 1: 10}, random_state=42)


# In[39]:


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20]
}

grid_search = GridSearchCV(best_tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[40]:


# Best hyperparameters
print(f"Best parameters: {grid_search.best_params_}")


# In[41]:


# Fitting the model with the best parameters
best_tree = grid_search.best_estimator_
best_tree.fit(X_train, y_train)


# In[42]:


# Making predictions on the test set
y_pred = best_tree.predict(X_test)


# In[43]:


# Evaluating the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:\n", cm)


# In[ ]:


# Visualizing the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=['<=50K', '>50K'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()


# In[ ]:




