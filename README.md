# Designing Predictive Models for Hospital Readmissions

This repository contains code and analysis for developing predictive models to prevent hospital readmissions. The project applies statistical learning and machine learning techniques to a large Kaggle dataset of hospital records, aiming to identify key factors associated with readmissions and to build predictive models that can help reduce healthcare costs and improve patient outcomes.

**Overview**

**Motivation:** Hospital readmissions account for billions of dollars in healthcare costs annually. In 2018 alone, 3.8 million adult hospital readmissions occurred within 30 days in the U.S., with an average cost of $15,200 per readmission.

**Objective:** Analyze 10 years of patient records to predict the likelihood of readmission and identify influential factors.

**Approach:** Compare classical statistical models with modern machine learning algorithms to balance interpretability and predictive power.

**Dataset:**
  The dataset is available on Kaggle: [Download the Hospital Readmissions Dataset on Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download)
  Please download it manually from Kaggle and place it in the project folder as: hospital_readmissions.csv.

**Methodologies:**
   - The following techniques were applied:
       - Exploratory Data Analysis (EDA)
       - Correlation heatmaps
       - Age, hospital stay, and comorbidity patterns
       - Statistical Models
       - Logistic regression (full and stepwise selection)
       - Parsimonious GLM (with selected key predictors)
       - Regularization Methods (via glmnet)
       - Ridge regression
       - LASSO regression
       - Elastic Net
       - Machine Learning
       - Random Forest (ensemble model)
    
**Evaluation**
  - Models are compared using:
      - Accuracy, Precision, Recall, and F1-score
      - ROC curves and AUC values
      - Confusion matrices
   
**Author:**
  - **Developed by Behnam Jabbari Marand, Ph.D. Student, NC State University**
  - Focus: Optimization, integer programming, and power systems applications.
