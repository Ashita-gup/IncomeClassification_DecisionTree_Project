#  Income Classification using Decision Tree

This project focuses on predicting whether an individual earns more than $50K per year using demographic and work-related features. A Decision Tree Classifier is used to learn patterns from the Adult Census Income dataset and make predictions. The goal of this project is not just to build a model, but to understand the complete machine learning workflow from raw data to evaluation.

---

##  Project Overview

The task is a binary classification problem where the model categorizes individuals into:
- <=50K
- >50K

The model uses features such as age, education, occupation, working hours, and more to make predictions about income levels.

---

##  Methodology

The project follows a clear step-by-step machine learning pipeline:

### Data Loading
The dataset is loaded and column names are assigned to make the data easier to understand and work with.

### Data Cleaning
Missing values are removed to ensure the model is trained on consistent and reliable data.

### Encoding Categorical Data
Categorical features are converted into numerical values using Label Encoding, since machine learning models cannot directly process text.

### Feature and Target Separation
- Features (X): All input variables
- Target (y): Income category

### Train-Test Split
The dataset is split into 70% training data and 30% testing data to evaluate how well the model performs on unseen data.

### Model Building
A Decision Tree Classifier is used to model the data. Class weights are applied to handle imbalance in the dataset so that the model does not ignore less frequent classes.

### Hyperparameter Tuning
GridSearchCV is used to find the best combination of parameters such as:
- Splitting criterion (gini or entropy)
- Maximum depth of the tree
- Minimum samples required to split

This step improves the overall performance of the model.

### Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics provide a detailed understanding of how well the model is performing.

### Visualization
The final decision tree is visualized to understand how decisions are being made by the model, making it more interpretable.

---

##  Project Structure

IncomeClassification_DecisionTreeProject/
│
├── IncomeClassification_DecisionTree.py
├── IncomeClassification_DecisionTree_Project.ipynb
├── README.md

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

##  Key Highlights

- Built a complete end-to-end machine learning pipeline
- Handled missing and categorical data effectively
- Addressed class imbalance using class weights
- Improved model performance using GridSearchCV
- Evaluated the model using multiple metrics
- Visualized the decision tree for better interpretability

---

##  How to Run

1. Clone the repository:
   git clone https://github.com/Ashita-gup/IncomeClassification_DecisionTree_Project.git

2. Navigate to the project folder:
   cd IncomeClassification_DecisionTreeProject

3. Run the Python file:
   python IncomeClassification_DecisionTree.py

You can also open the Jupyter Notebook to go through the implementation step by step.

---

##  Conclusion

This project shows how a Decision Tree can be used to solve a real-world classification problem. It highlights the importance of data preprocessing, model tuning, and evaluation. It also demonstrates how models can be made interpretable using visualization techniques.

---

##  Author

Ashita Gupta  
B.Tech in CSE with specialization in Cyber Security
