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
@st.cache_data
def load_data():
    df = pd.read_csv('Train_bigmart.csv')
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    mode = df.pivot_table(values="Outlet_Size",columns="Outlet_Type",aggfunc=lambda x: x.mode()[0])
    missing = df['Outlet_Size'].isnull()
    df.loc[missing, 'Outlet_Size'] = df.loc[missing, 'Outlet_Type'].apply(lambda x: mode[x][0])
    df['Outlet_Size'] = df['Outlet_Size'].astype(str)
    df.replace({"Item_Fat_Content": {'low fat': 'Low Fat','LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
    return df
data = load_data()
original_data = data.copy()
@st.cache_resource
def encode_data(df):
    encoders = {}
    for col in ["Item_Identifier", "Item_Fat_Content", "Item_Type","Outlet_Identifier", "Outlet_Size","Outlet_Location_Type", "Outlet_Type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders
data, encoder_dict = encode_data(data)
@st.cache_resource
def train_model(df):
    X = df.drop("Item_Outlet_Sales", axis=1)
    Y = df["Item_Outlet_Sales"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    return model, X, X_train, X_test, Y_train, Y_test
model, X, X_train, X_test, Y_train, Y_test = train_model(data)
st.subheader("Model Performance")
st.write("Train R2:", metrics.r2_score(Y_train, model.predict(X_train)))
st.write("Test R2:", metrics.r2_score(Y_test, model.predict(X_test)))
st.subheader("Data Visualization")
if st.checkbox("Show Item MRP Distribution"):
    fig, ax = plt.subplots()
    sns.histplot(original_data['Item_MRP'], kde=True, ax=ax)
    st.pyplot(fig)
if st.checkbox("Show Outlet Type Count"):
    fig, ax = plt.subplots()
    sns.countplot(x='Outlet_Type', data=original_data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
st.subheader("Enter Product Details")
def user_input():
    return { "Item_Identifier": st.selectbox("Item Identifier", original_data['Item_Identifier'].unique()), "Item_Weight": st.number_input("Item Weight", min_value=0.0), "Item_Fat_Content": st.selectbox("Item Fat Content", original_data['Item_Fat_Content'].unique()), "Item_Visibility": st.number_input("Item Visibility", min_value=0.0), "Item_Type": st.selectbox("Item Type", original_data['Item_Type'].unique()), "Item_MRP": st.number_input("Item MRP", min_value=0.0),  "Outlet_Identifier": st.selectbox("Outlet Identifier", original_data['Outlet_Identifier'].unique()), "Outlet_Establishment_Year": st.number_input("Outlet Establishment Year", min_value=1980, max_value=2025), "Outlet_Size": st.selectbox("Outlet Size", original_data['Outlet_Size'].unique()), "Outlet_Location_Type": st.selectbox("Outlet Location Type", original_data['Outlet_Location_Type'].unique()), "Outlet_Type": st.selectbox("Outlet Type", original_data['Outlet_Type'].unique())}
input_dict = user_input()
error = False
for col in encoder_dict:
    try:
        input_dict[col] = encoder_dict[col].transform([input_dict[col]])[0]
    except:
        st.error(f"Invalid value for {col}")
        error = True
if not error:
    input_data = np.array([input_dict[col] for col in X.columns]).reshape(1, -1)
    if st.button("Predict Sales"):
        pred = model.predict(input_data)
        st.success(f"Predicted Sales: ₹ {pred[0]:.2f}")









