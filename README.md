# Fraud Detection

## Overview
This project focuses on detecting fraudulent transactions using a highly imbalanced credit card fraud dataset. The data has been preprocessed and explored, and various models will be trained and evaluated for their effectiveness in identifying frauds.    
[View this notebook in kaggle](https://www.kaggle.com/code/arunl15/recall-precision-optimization)

## Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation. 
This is done to keep the features and the values confidential. Therefore Features `V1`, `V2`, … `V28` are the principal components obtained with PCA. The only features which have not been transformed with PCA are `Time` and `Amount`.   
[Click here to view the dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## EDA Keys Steps

### Data Overview and Initial Analysis
- The dataset consists of 284,807 entries and 30 features.
- There are no null values in the dataset.
- There are 1,081 duplicate entries and all of them are removed.

### Data Visualization & Feature Extraction
- Used FacetGrid to plot histograms with density statistics and KDE (Kernel Density Estimate) overlays to compare the distribution of each variable between the fraudulent and authentic classes.
- Plotted the distribution of the Time feature, differentiating between classes.
- Converted the Time feature from seconds to HH:MM:SS format and removed the original `Time` column from the dataset. This was done to potentially enhance the model's discriminative power, especially if fraudulent transactions tend to occur at specific times.
- Created boxplots for each feature to examine the presence of outliers. Significant number of outliers are present but decided to retain outliers as they represent authentic transactional data.

## Model Training Key Steps
- Training a model with undersamples was ruled out since that results is huge loss in valuable data and such a model performs well solely on the undersampled test set, but when evaluated on the real test set, it generates numerous false positives resulting in very low precision for the fraud class which is not desirable.
- `The goal of this projects was to train a model that prioritizes high recall while maintaining an acceptable precision level, as false negatives in this scenario are much more critical and can lead to significant financial losses.`
- Several models were trained using XGBoost Classifiers using Original Non-Resampled data and also Resampled data using resampling methods like ADASYN and BorderlineSMOTE.
- Optuna was used for optimizing these models by trying to miximize the AUCPR (area under the precision-recall curve).
- Multi Objective optimization was also tried for the models by maximizing both Precision and Recall simultaneously as much as possible and getting the solutions on the Pareto front (best Precision-Recall trade-off's)
- Out of all the models, the best performing one, was the model trained on the ADASYN resampled dataset and optimized for maximizing AUCPR using Optuna.
- Precision-Recall Curve and the ROC Curve of the best model was plotted.
- The `precision-recall values and confusion matrix plot of this model at threshold values of 0.05, 0.10, ..., 0.95 were also plotted.` This demonstrates how we can use this same model with an appropriate threshold set by us to achieve the desired precision and recall values.
