import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
st.title("Big Mart Sales Prediction App")
big_mart_data = pd.read_csv('Train_bigmart.csv')
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
mode_of_outlet_size = big_mart_data.pivot_table(values = "Outlet_Size", columns = "Outlet_Type", aggfunc = (lambda x: x.mode()[0]))
missing_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])
big_mart_data.replace({"Item_Fat_Content": {'low fat' : 'Low Fat', 'LF' : 'Low Fat', 'reg' : 'Regular'}}, inplace=True)
original_data = big_mart_data.copy()
encoder_dict = {}
for col in ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
  le = LabelEncoder()
  big_mart_data[col] = le.fit_transform(big_mart_data[col])
  encoder_dict[col] = le
X = big_mart_data.drop(columns = "Item_Outlet_Sales", axis=1)
Y = big_mart_data['Item_Outlet_Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
regressor=XGBRegressor()
regressor.fit(X_train,Y_train)
training_data_prediction=regressor.predict(X_train)
st.write("Model Performance")
r2_train=metrics.r2_score(Y_train,training_data_prediction)
st.write("R squared value on training data = ",r2_train)
test_data_prediction=regressor.predict(X_test)
r2_test=metrics.r2_score(Y_test,test_data_prediction)
st.write("R squared value on test data = ",r2_test)
st.subheader("Data Visualization")
if st.checkbox("Show Distributions"):
    fig, ax = plt.subplots()
    sns.histplot(original_data['Item_MRP'], kde=True, ax=ax)
    st.pyplot(fig)
if st.checkbox("Show Outlet Type Count"):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x='Outlet_Type', data=original_data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
st.subheader("Enter Product Details")
def user_input():
    Item_Identifier = st.selectbox("Item Identifier",original_data['Item_Identifier'].unique())
    Item_Weight = st.number_input("Item Weight", value=9.3)
    Item_Fat_Content = st.selectbox("Item Fat Content",original_data['Item_Fat_Content'].unique())
    Item_Visibility = st.number_input("Item Visibility", value=0.016)
    Item_Type = st.selectbox("Item Type",original_data['Item_Type'].unique())
    Item_MRP = st.number_input("Item MRP", value=249.8)
    Outlet_Identifier = st.selectbox("Outlet Identifier",original_data['Outlet_Identifier'].unique())
    Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", value=1999)
    Outlet_Size = st.selectbox("Outlet Size", original_data['Outlet_Size'].unique())

    Outlet_Location_Type = st.selectbox(
        "Outlet Location Type",
        original_data['Outlet_Location_Type'].unique()
    )

    Outlet_Type = st.selectbox(
        "Outlet Type",
        original_data['Outlet_Type'].unique()
    )

    return {
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_Weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_Visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Establishment_Year": Outlet_Establishment_Year,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type
    }











