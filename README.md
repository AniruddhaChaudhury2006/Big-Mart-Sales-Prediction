# Big Mart Sales Prediction

This project aims to predict the sales of various products across different Big Mart outlets. It involves comprehensive data analysis, preprocessing, feature engineering, and the development of a machine learning model using XGBoost to forecast `Item_Outlet_Sales`.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features and Target](#features-and-target)
- [Libraries Used](#libraries-used)
- [Code Structure and Methodology](#code-structure-and-methodology)
- [Results](#results)

## Project Overview

The goal of this project is to build a predictive model that can accurately estimate the sales of items in different Big Mart stores. The process includes:
1.  **Data Loading and Initial Exploration**: Understanding the dataset's structure and identifying initial data quality issues.
2.  **Data Preprocessing**: Handling missing values and cleaning inconsistencies.
3.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships to gain insights.
4.  **Feature Engineering**: Transforming categorical variables into numerical representations suitable for machine learning.
5.  **Model Training**: Implementing an XGBoost Regressor model.
6.  **Model Evaluation**: Assessing the model's performance using appropriate metrics.
7.  **Prediction**: Demonstrating how to use the trained model for new predictions.

## Dataset

The dataset used for this project is `Train_bigmart.csv`. It contains historical sales data for various products sold in Big Mart stores.

## Features and Target

**Key Features:**
-   `Item_Identifier`: Unique product ID
-   `Item_Weight`: Weight of the product
-   `Item_Fat_Content`: Fat content of the product (e.g., Low Fat, Regular)
-   `Item_Visibility`: Percentage of total display area allocated to the product in the store
-   `Item_Type`: Category of the product
-   `Item_MRP`: Maximum Retail Price (list price) of the product
-   `Outlet_Identifier`: Unique store ID
-   `Outlet_Establishment_Year`: Year in which the store was established
-   `Outlet_Size`: Size of the store (e.g., Small, Medium, High)
-   `Outlet_Location_Type`: Type of city where the store is located (e.g., Tier 1, Tier 2, Tier 3)
-   `Outlet_Type`: Type of the outlet (e.g., Supermarket Type1, Grocery Store)

**Target Variable:**
-   `Item_Outlet_Sales`: Sales of the product in a particular store (this is the variable to be predicted)

## Libraries Used

-   `numpy`: For numerical operations.
-   `pandas`: For data manipulation and analysis.
-   `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
-   `seaborn`: For high-level statistical data visualization.
-   `sklearn.preprocessing.LabelEncoder`: For encoding categorical features.
-   `sklearn.model_selection.train_test_split`: For splitting data into training and testing sets.
-   `xgboost.XGBRegressor`: For building the predictive regression model.
-   `sklearn.metrics`: For model evaluation metrics.

## Code Structure and Methodology

The notebook follows a standard machine learning workflow:

1.  **Initial Setup & Data Loading**:
    -   Imports necessary libraries.
    -   Loads the `Train_bigmart.csv` dataset into a pandas DataFrame.
    -   Displays the first few rows and general information about the data.

2.  **Handling Missing Values**:
    -   Calculates the mean of `Item_Weight` and fills missing values with this mean.
    -   Identifies the mode of `Outlet_Size` based on `Outlet_Type` for each outlet type.
    -   Fills missing `Outlet_Size` values using the calculated modes.
    -   Verifies that all missing values have been addressed.

3.  **Exploratory Data Analysis (EDA)**:
    -   Generates descriptive statistics for numerical columns.
    -   Visualizes the distribution of key numerical features like `Item_Weight`, `Item_Visibility`, `Item_MRP`, and `Item_Outlet_Sales` using histograms/KDE plots (`sns.distplot`).
    -   Analyzes the counts of categorical features such as `Outlet_Establishment_Year`, `Item_Fat_Content`, `Item_Type`, and `Outlet_Size` using count plots (`sns.countplot`).

4.  **Data Cleaning & Feature Engineering**:
    -   Standardizes the `Item_Fat_Content` column by mapping 'low fat', 'LF' to 'Low Fat', and 'reg' to 'Regular'.
    -   Applies `LabelEncoder` to transform all identified categorical features (`Item_Identifier`, `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`) into numerical labels.

5.  **Model Training**:
    -   Separates the dataset into features (`X`) and the target variable (`Y`).
    -   Splits the data into training (80%) and testing (20%) sets using `train_test_split` with a `random_state` for reproducibility.
    -   Initializes an `XGBRegressor` model.
    -   Trains the `XGBRegressor` model on the training data.

6.  **Model Evaluation**:
    -   Makes predictions on both the training and test datasets.
    -   Calculates and prints the R-squared score for both training and test data to assess model performance and identify potential overfitting.

7.  **Making Predictions**:
    -   Demonstrates how to use the trained model to predict `Item_Outlet_Sales` for a new input data point.
